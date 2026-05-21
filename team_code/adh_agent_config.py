class GlobalConfig:
    """base architecture configurations"""

    # PID
    turn_KP         = 1.25
    turn_KI         = 0.75
    turn_KD         = 0.3
    turn_n          = 40        # buffer size
    
    speed_KP        = 5.0
    speed_KI        = 0.5
    speed_KD        = 1.0
    speed_n         = 40       # buffer size

    max_throttle    = 0.75 # upper limit on throttle signal value in dataset
    brake_speed     = 0.1  # desired speed below which brake is triggered
    brake_ratio     = 1.1  # ratio of speed to desired speed at which brake is triggered
    clip_delta      = 0.35 # maximum change in speed input to logitudinal controller

    # Planner
    planner_device                      ='cuda:0'
    planner_model_ckpt                  = "./checkpoints/llava15-ours"
    planner_load_8bit                   = False
    planner_load_4bit                   = True
    
    # Controller
    controller_device                   = 'cuda:0'
    controller_preception_model         = 'memfuser_baseline_e1d3_return_feature'
    controller_preception_model_ckpt    = './checkpoints/vision_weights/vision-encoder-r50.pth.tar'
    controller_model                    = './checkpoints/opt-350m-ours'    
    controller_model_ckpt               = './checkpoints/opt-350m-ours.pth' # my own training weights    

    has_motion              = True
    has_llava_vision_tower  = False
    agent_use_notice        = False
    disable_planner         = False     # if disable it, just use the actor to achieve the control
    use_api_planner         = True      # if disable it, we will load the planner model locally, it needs a lot of memory.
    use_previous_motion     = False     # this means whether use previous motion, if does we will send previous as a sentence to the planner and actor.
    sample_rate             = 2

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
