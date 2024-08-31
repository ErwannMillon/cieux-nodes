from .nodes import SamplerDEIS

NODE_CLASS_MAPPINGS = {
    "SamplerDEIS": SamplerDEIS,

}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SamplerDEIS": "Sampler DEIS",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
