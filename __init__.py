from .nodes import SamplerDEIS, FlowMatchScheduler

NODE_CLASS_MAPPINGS = {
    "SamplerDEIS": SamplerDEIS,
    "FlowMatchScheduler": FlowMatchScheduler,

}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SamplerDEIS": "Sampler DEIS",
    "FlowMatchScheduler": "FlowMatch Scheduler",

}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
