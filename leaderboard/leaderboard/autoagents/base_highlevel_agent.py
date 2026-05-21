from __future__ import print_function


# TODO more models are needed!
MODEL_LIST = ["gpt4v"]

class Base_highlevel_agent:
    """
    Base class for high-level agents
    inputs:
        inputs: dict stores the input data (rgb, text, lidar...)
        model_name: model names
    functions
        step: generate the output text
    """
    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name
        assert self.model_name in MODEL_LIST

    def step(self, inputs):
        """
        inputs: 
            input_text: the instruction
            image_path: the path of image
        outputs:
            the caption
        """
        # GPT4v inference code
        if self.model_name == "gpt4v":
            from utils import generate_query # prepare gpt4v inference function
            ori_instruction = inputs["input_text"]
            image_path = inputs["image_path"]
            prompt = f"""
                Your task is to determine what the safest and most efficient vehicle driving instruction is in the current environment. Your instruction should be based on the instruction: {ori_instruction}. 
                Your instruction does not need to be too complicated. Only return the instruction.
                """
            new_instruction = generate_query("gpt-4-vision-preview", image_path, prompt)
            return new_instruction 
        else:
            raise NotImplementedError
    





    
    


        



