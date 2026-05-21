# from lavis.common.registry import registry
# model_cls = registry.get_model_class('vicuna_drive')
# import pdb; pdb.set_trace()


import timm
timm.create_model('memfuser_baseline_e1d3_return_feature')
