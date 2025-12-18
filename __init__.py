from .samplers import (SamplerCustomAdvanced_Efficient,
                       SamplerCustomUltraAdvancedEfficient)


NODE_CLASS_MAPPINGS = {
    "SamplerCustomAdvanced_Efficient": SamplerCustomAdvanced_Efficient,
    "SamplerCustomUltraAdvancedEfficient": SamplerCustomUltraAdvancedEfficient,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SamplerCustomAdvanced_Efficient": "Sampler Custom Advanced (Efficient)",
    "SamplerCustomUltraAdvancedEfficient": "Sampler Custom Ultra Advanced (Efficient)",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']