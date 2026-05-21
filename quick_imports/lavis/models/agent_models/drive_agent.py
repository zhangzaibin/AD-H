"""
Requires Transformer 4.28 and above, implementation may change according the Llama implementation
"""
import logging
import string
from packaging import version

import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn
import torch.nn.functional as F

import transformers
import peft
from peft import LoraConfig, get_peft_model

from lavis.common.registry import registry
from lavis.models.drive_models.drive import Blip2VicunaDrive
# from lavis.models.agent_models.clip_encoder import CLIPVisionTower
# from lavis.models.agent_models.mm_projector import build_vision_projector
from timm import create_model
import yaml


def load_yaml(file_path):
    with open(file_path, 'r') as file:
        data = yaml.load(file, Loader=yaml.FullLoader)
    return data

@registry.register_model("drive_agent")
class DriveAgent(Blip2VicunaDrive):
    """
    BLIP2 Vicuna model.
    Supported model types:
        - vicuna7b
        - vicuna13b
    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("blip2_vicuna_instruct", "vicuna7b")
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "vicuna7b": "configs/models/blip2/blip2_instruct_vicuna7b.yaml",
        "vicuna13b": "configs/models/blip2/blip2_instruct_vicuna13b.yaml",
    }

    def __init__(
        self,
        preception_model="",
        preception_model_ckpt='',
        load_pretrained=True,
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        llm_model="",
        max_txt_len=128,
        use_extra_prompt=False,
        use_notice_prompt=False,
        freeze_decoder_of_visual_encoder=True,
        has_qformer=True,
        has_gru_decoder=False,
        has_lora=False,
        has_motion=True,
        has_llava_vision_tower=False,
        vision_tower_cfg={},
        mm_projector_cfg={},
        split_section_num_for_visual_encoder=2, # save gpu memory
    ):
        super().__init__(
            preception_model=preception_model,
            preception_model_ckpt=preception_model_ckpt,
            load_pretrained=load_pretrained,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            llm_model=llm_model,
            max_txt_len=max_txt_len,
            use_extra_prompt=use_extra_prompt,
            use_notice_prompt=use_notice_prompt,
            freeze_decoder_of_visual_encoder=freeze_decoder_of_visual_encoder,
            has_qformer=has_qformer,
            has_gru_decoder=has_gru_decoder,
            has_lora=has_lora,
            split_section_num_for_visual_encoder=split_section_num_for_visual_encoder,
        )
        
        self.llm_model_name = llm_model
        self.has_motion = has_motion # whether to adopt motion into the imput prompt
        if self.has_motion:
            print("#############enable motion setting###########")
        self.has_llava_vision_tower = has_llava_vision_tower # whether use the llava vision tower to introduce extra parameters.

        if self.has_llava_vision_tower:
            # llava setting. TODO add a trigger
            # load the clip vision tower
            self.vision_tower = CLIPVisionTower(vision_tower_cfg['vision_tower'], args=vision_tower_cfg)

            # load mm_projector
            if getattr(self, 'mm_projector', None) is None:
                self.mm_projector = build_vision_projector(mm_projector_cfg)

                if 'unpad' in mm_projector_cfg["mm_patch_merge_type"]:
                    embed_std = 1 / torch.sqrt(torch.tensor(mm_projector_cfg["hidden_size"], dtype=self.dtype))
                    self.image_newline = nn.Parameter(
                        torch.randn(mm_projector_cfg["hidden_size"], dtype=self.dtype) * embed_std
                    )
            else:
                # In case it is frozen by LoRA
                for p in self.mm_projector.parameters():
                    p.requires_grad = True

            if mm_projector_cfg["pretrain_mm_mlp_adapter"] is not None:
                mm_projector_weights = torch.load(mm_projector_cfg["pretrain_mm_mlp_adapter"], map_location='cpu')
                def get_w(weights, keyword):
                    return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

                self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))
        else:
            print("disable llava vision tower setting")

        # BUG opt-350 has some issues so we need to adapt it
        if 'opt-350' in self.llm_model_name:
            print('using opt-350m')
            self.llm_model.config.hidden_size = 512
            self.llm_proj = nn.Linear(self.Qformer.config.hidden_size, self.llm_model.config.hidden_size)
            self.prompt_proj_for350 = nn.Linear(512, 512)
            self.waypoints_predictor = nn.Sequential(
                            nn.Linear(512, 512),
                            nn.ReLU(),
                            nn.Linear(512, 10)
            )
            self.end_predictor = nn.Sequential(
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512, 2)
        )
            
    def encode_images(self, images):
        """
        encode the images for llava vision tower
        """
        image_features = self.vision_tower(images)
        image_features = self.mm_projector(image_features)
        # TODO support multi-image processing
        return image_features

    def motion_prompt(self, instruction, motion):
        """
        This fuction is to generate prompt to the actor model.
        Args:
            instruction: `list of str` long-horizon goal like "turn left at the end of the road"
            motion: `list of str` low-level instruction like "apply brakes"
        
        Returns:
            what action should the car do to <instruction> with motion <motion>
        """
        motion_prompt_list = []
        assert len(instruction) == len(motion)
        for index in range(len(instruction)):
            if self.has_motion:
                motion_prompt_list.append(f"What action should the car do to <{instruction[index]}> with the perception and motion <{motion[index]}>?")
                #motion_prompt_list.append(motion[index])
            else:
                motion_prompt_list.append(f"What action should the car do to <{instruction[index]}>?")

        return motion_prompt_list

    
    def forward(self, samples, inference_mode=False, image_embeds=None):
        if image_embeds is None: # train mode
            device = samples["rgb_front"].device
            bs = samples['rgb_front'].size(0)
            t = samples['rgb_front'].size(1)
            for key in ['rgb_front', 'rgb_left', 'rgb_right', 'rgb_rear', 'rgb_center', 'lidar', 'num_points', 'velocity']:
                shapz = samples[key].size()
                samples[key] = samples[key].view(bs*t, *shapz[2:])

            if self.freeze_decoder_of_visual_encoder:
                with torch.no_grad():
                    with self.maybe_autocast():
                        image_embeds_full = []
                        splited_samples = self.split_data(samples)
                        for i in range(self.split_section_num_for_visual_encoder):
                            image_embeds = self.visual_encoder(splited_samples[i])
                            image_embeds_full.append(image_embeds)
                        image_embeds = torch.cat(image_embeds_full, dim=0)
            else:
                with self.maybe_autocast():
                    image_embeds = self.visual_encoder(samples)
        else: # inference mode
            device = image_embeds.device
            bs = image_embeds.size(0)
            t = image_embeds.size(1)
            image_embeds = image_embeds.view(bs*t, *image_embeds.size()[2:])

        image_embeds = self.ln_vision(image_embeds) # layernorm

        
        if self.has_llava_vision_tower:
            #prepare clip image embeds
            # for now we freeze the clip and mm injector 
            image_front = samples["rgb_front"] # for the clip tower, we only use rgb front TODO support multi frame input
            # resize the image tensor 224 --> 336 (llava's resolution)
            image_front = F.interpolate(image_front, size=(336, 336), mode='bilinear', align_corners=False) # batch_size 3 336 336
            image_clip_feature = self.encode_images(image_front) # clip freeze mm_projector unfreeze

        # perpare the text input with instruction and motion
        # instruction --> samples["text_input"]
        # motion --> samples["motion"]
        motion_prompt = self.motion_prompt(samples["text_input"], samples["motion"])

        if self.has_qformer:
            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
            text_Qformer = self.llm_tokenizer(
                [i for i in motion_prompt for _ in range(t)],
                padding='longest',
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(device)
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(device)
            try:
                Qformer_atts = torch.cat([query_atts, text_Qformer.attention_mask],dim=1)
            except:
                # BUG I do not know why, in the last round of evaluation raise error.
                # TODO
                # _d1 = text_Qformer.attention_mask.size(0)
                # _d2 = query_tokens.size(1)
                # query_atts = torch.ones((_d1, _d2), dtype=torch.long).to(device)
                # Qformer_atts = torch.cat([query_atts, text_Qformer.attention_mask],dim=1)
                print(query_atts.size(), text_Qformer.attention_mask.size())
                print(f"the first size of image_embeds is {image_embeds.shape[0]}")
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device)

            query_output = self.Qformer.bert(
                text_Qformer.input_ids,
                attention_mask=Qformer_atts,
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

        image_embeds = self.llm_proj(query_output.last_hidden_state[:,:query_tokens.size(1),:])

        image_embeds = image_embeds.view(bs, t, *image_embeds.size()[1:])

        if self.has_llava_vision_tower:
            # clip embeds
            clip_embeds = image_clip_feature.view(bs, t, *image_clip_feature.size()[1:])
            # cat image_embeds and clip_embeds
            image_embeds = torch.cat((image_embeds, clip_embeds), dim=2)

        if self.use_extra_prompt:
            text_before_img = samples['text_before_img']
            text_after_img = samples['text_after_img']
            image_embeds, image_atts, end_flag_pos_list = self.prompt_wrap(image_embeds, text_before_img, text_after_img, samples['valid_frames'])
        else:
            image_atts = None
            end_flag_pos_list = []
            n_length = image_embeds.size(2) # token number for each frame
            for i in range(bs):
                end_flag_pos_list.append([n_length*(j+1)-1 for j in range(samples['valid_frames'][i])])

        self.llm_tokenizer.padding_side = "right"
        self.llm_tokenizer.truncation_side = 'left'
        text_input_tokens = self.llm_tokenizer(
            motion_prompt,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
        ).to(device)

        inputs_embeds = self.llm_model.get_input_embeddings()(text_input_tokens.input_ids)
        # inputs_embeds shape: (batch_size, sequence_length, hidden_size)
        # BUG fix the opt-350m TODO
        if 'opt-350' in self.llm_model_name:
            inputs_embeds = self.prompt_proj_for350(inputs_embeds)

        if self.use_notice_prompt:
            llm_inputs, llm_attention_mask, input_part_targets_len, wp_target_index = self.concat_text_image_input_with_notice(inputs_embeds, text_input_tokens.attention_mask,
                                                                                                                   image_embeds, samples['valid_frames'], end_flag_pos_list,
                                                                                                                   samples['notice_frame_id'], samples['notice_text'], image_atts)
        else:
            llm_inputs, llm_attention_mask, input_part_targets_len, wp_target_index = self.concat_text_image_input(inputs_embeds, text_input_tokens.attention_mask,
                                                                                                                   image_embeds, samples['valid_frames'], end_flag_pos_list, image_atts)
        wp_target_index = torch.tensor(wp_target_index, device=device).long()

        
        if "opt" in self.llm_model_name:
            with self.maybe_autocast():
                hidden_states = self.llm_model(
                    inputs_embeds=llm_inputs,
                    attention_mask=llm_attention_mask,
                    return_dict=False,
                ) # hidden states and logits
        else:
            with self.maybe_autocast():
                hidden_states = self.llm_model(
                    inputs_embeds=llm_inputs,
                    attention_mask=llm_attention_mask,
                    return_dict=False,
                ) # BUG there is no hidden stage in inference stage.

        # predicted_waypoints: bs, seq_len, 10
        if self.has_gru_decoder:
            output_wp = []
            _, n_tokens, _ =hidden_states.size()
            x = torch.zeros(size=(bs*n_tokens, 2), dtype=hidden_states.dtype).to(device)
            target_point = samples['target_point'].view(bs, -1, 2).to(device)

            target_point_list = []
            for i in range(bs):
                target_point_list.append(target_point[i, :samples['valid_frames'][i], :])
            target_point = torch.cat(target_point_list, 0)


            target_point_zeros = torch.zeros(size=(bs, n_tokens, 2), dtype=hidden_states.dtype).to(device)
            target_point_zeros[wp_target_index[:,0], wp_target_index[:, 1]] = target_point.to(hidden_states.dtype)
            target_point_zeros = target_point_zeros.view(bs*n_tokens, 2)
            target_point = target_point_zeros

            waypoints_feature = self.waypoints_fc(hidden_states.reshape(-1, self.llm_model.config.hidden_size))
            for _ in range(5):
                x_in = x# + target_point
                waypoints_feature = self.waypoints_predictor(x_in, waypoints_feature)
                dx = self.waypoints_output(waypoints_feature)
                x = dx + x
                output_wp.append(x)
            predicted_waypoints = torch.cat(output_wp, dim=1)
            predicted_waypoints = predicted_waypoints.view(bs, n_tokens, 10)

        else:
            predicted_waypoints = self.waypoints_predictor(hidden_states)
            # predicted_waypoints: N * 10
        predicted_waypoints = predicted_waypoints[wp_target_index[:,0], wp_target_index[:, 1]]
        predicted_end_prob = self.end_predictor(hidden_states)
        predicted_end_prob = predicted_end_prob[wp_target_index[:,0], wp_target_index[:, 1]]
        if inference_mode:
            return predicted_waypoints, predicted_end_prob

        gt_waypoints = self.build_gt_waypoints(samples['local_future_waypoints'], samples['valid_frames'])
        waypoints_loss = self.waypoints_loss(predicted_waypoints, gt_waypoints)

        gt_end_flags = self.build_gt_end_flags(samples['valid_frames'])
        end_loss = self.end_loss(predicted_end_prob, gt_end_flags)

        predicted_end = torch.argmax(predicted_end_prob, dim=1)
        end_acc = (predicted_end == gt_end_flags).float().mean().item()

        loss = waypoints_loss + end_loss * 0.2

        return {"loss": loss, 'waypoints_loss': waypoints_loss, 'end_loss': end_loss, 'end_acc': end_acc}
    

    @classmethod
    def from_config(cls, cfg):
        preception_model = cfg.get("preception_model")
        preception_model_ckpt = cfg.get("preception_model_ckpt")
        load_pretrained = cfg.get('load_pretrained', True)
        img_size = cfg.get("image_size")
        llm_model = cfg.get("llm_model")

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)

        max_txt_len = cfg.get("max_txt_len", 64)
        use_extra_prompt = cfg.get("use_extra_prompt", False)
        use_notice_prompt = cfg.get("use_notice_prompt", False)
        freeze_decoder_of_visual_encoder = cfg.get("freeze_decoder_of_visual_encoder", True)
        has_gru_decoder = cfg.get("has_gru_decoder", False)
        has_lora = cfg.get('has_lora', False)
        has_motion = cfg.get('has_motion', True)
        has_llava_vision_tower = cfg.get('has_llava_vision_tower', False)
        vision_tower_cfg = cfg.get('vision_tower_cfg', {})
        mm_projector_cfg = cfg.get('mm_projector_cfg', {})


        split_section_num_for_visual_encoder = cfg.get('split_section_num_for_visual_encoder', 2)

        model = cls(
            img_size=img_size,
            preception_model=preception_model,
            preception_model_ckpt=preception_model_ckpt,
            load_pretrained=load_pretrained,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            llm_model=llm_model,
            max_txt_len=max_txt_len,
            use_extra_prompt=use_extra_prompt,
            use_notice_prompt=use_notice_prompt,
            freeze_decoder_of_visual_encoder=freeze_decoder_of_visual_encoder,
            has_gru_decoder=has_gru_decoder,
            has_lora=has_lora,
            has_motion=has_motion,
            has_llava_vision_tower=has_llava_vision_tower,
            vision_tower_cfg=vision_tower_cfg,
            mm_projector_cfg=mm_projector_cfg,
            split_section_num_for_visual_encoder=split_section_num_for_visual_encoder,
        )


        return model