import torch
import latent_preview
from PIL import Image

import comfy
from nodes import VAEDecodeTiled, PreviewImage, VAEDecode
from comfy_extras.nodes_custom_sampler import Noise_EmptyNoise, Noise_RandomNoise

from .utils import (pil2tensor, global_preview_method, warning, 
                   set_preview_method, store_ksampler_results, 
                   globals_cleanup, sample_custom_ultra)


class SamplerCustomAdvanced_Efficient:
    # Image Preview code taken from jags111's efficiency-nodes (TSC_KSampler)
    empty_image = pil2tensor(Image.new('RGBA', (1, 1), (0, 0, 0, 0)))
    
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"noise": ("NOISE", ),
                    "guider": ("GUIDER", ),
                    "sampler": ("SAMPLER", ),
                    "sigmas": ("SIGMAS", ),
                    "latent_image": ("LATENT", ),
                    "preview_method": (["auto", "latent2rgb", "taesd", "vae_decoded_only", "none"],),
                    "vae_decode": (["true", "true (tiled)", "false"],),
                     },
                     "optional": { "optional_vae": ("VAE",),
                    },
                    "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO", "my_unique_id": "UNIQUE_ID",},
                }

    RETURN_TYPES = ("LATENT", "LATENT", "VAE", "IMAGE",)
    RETURN_NAMES = ("latent", "denoised_latent", "VAE", "IMAGE",)
    FUNCTION = "sample"
    CATEGORY = "More Efficient Samplers"

    def sample(self, noise, guider, sampler, sigmas, latent_image, preview_method, vae_decode, optional_vae=(None,), prompt=None, extra_pnginfo=None, my_unique_id=None):
        latent = latent_image
        latent_image = latent["samples"]
        latent = latent.copy()
        latent_image = comfy.sample.fix_empty_latent_channels(guider.model_patcher, latent_image)
        latent["samples"] = latent_image

        noise_mask = None
        if "noise_mask" in latent:
            noise_mask = latent["noise_mask"]
        
        # Rename the vae variable
        vae = optional_vae
        # If vae is not connected, disable vae decoding
        if vae == (None,) and vae_decode != "false":
            print(f"{warning('KSampler(Efficient) Warning:')} No vae input detected, proceeding as if vae_decode was false.\n")
            vae_decode = "false"
        
        # ------------------------------------------------------------------------------------------------------
        def vae_decode_latent(vae, out, vae_decode):
            return VAEDecodeTiled().decode(vae,out,320)[0] if "tiled" in vae_decode else VAEDecode().decode(vae,out)[0]
        # ---------------------------------------------------------------------------------------------------------------

        def process_latents():
            x0_output = {}
            # Initialize output variables
            out = out_denoised = images = preview = previous_preview_method = None

            try:
                # Change the global preview method (temporarily)
                set_preview_method(preview_method)
                
                callback = latent_preview.prepare_callback(guider.model_patcher, sigmas.shape[-1] - 1, x0_output)
    
                disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
                samples = guider.sample(noise.generate_noise(latent), latent_image, sampler, sigmas, denoise_mask=noise_mask, callback=callback, disable_pbar=disable_pbar, seed=noise.seed)
                samples = samples.to(comfy.model_management.intermediate_device())

                out = latent.copy()
                out["samples"] = samples
                if "x0" in x0_output:
                    out_denoised = latent.copy()
                    out_denoised["samples"] = guider.model_patcher.model.process_latent_out(x0_output["x0"].cpu())
                else:
                    out_denoised = out

                previous_preview_method = global_preview_method()
                
                # ---------------------------------------------------------------------------------------------------------------
                # Decode image if not yet decoded
                if "true" in vae_decode:
                    if images is None:
                        images = vae_decode_latent(vae, out, vae_decode)
                        # Store decoded image as base image of no script is detected
                        store_ksampler_results("image", my_unique_id, images)

                # Define preview images
                if preview_method == "none" or (preview_method == "vae_decoded_only" and vae_decode == "false"):
                    preview = {"images": list()}
                elif images is not None:
                    preview = PreviewImage().save_images(images, prompt=prompt, extra_pnginfo=extra_pnginfo)["ui"]

                # Define a dummy output image
                if images is None and vae_decode == "false":
                    images = SamplerCustomAdvanced_Efficient.empty_image
            
            finally:
                # Restore global changes
                set_preview_method(previous_preview_method)
              
            return preview, out, out_denoised, images
        # ---------------------------------------------------------------------------------------------------------------
        # Clean globally stored objects of non-existant nodes
        globals_cleanup(prompt)
        # ---------------------------------------------------------------------------------------------------------------
        preview, out, out_denoised, images = process_latents()

        result = (out, out_denoised, vae, images,)

        if preview is None:
            return {"result": result}
        else:
            return {"ui": preview, "result": result}
        

class SamplerCustomUltraAdvancedEfficient:
    # Image Preview code taken from jags111's efficiency-nodes (TSC_KSampler)
    empty_image = pil2tensor(Image.new('RGBA', (1, 1), (0, 0, 0, 0)))

    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"model": ("MODEL",),
                    "add_noise": ("BOOLEAN", {"default": True}),
                    "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
                    "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01}),
                    "positive": ("CONDITIONING", ),
                    "negative": ("CONDITIONING", ),
                    "sampler": ("SAMPLER", ),
                    "sigmas": ("SIGMAS", ),
                    "latent_image": ("LATENT", ),
                    "start_at_step": ("INT", {"default": 0, "min": 0, "max": 10000}),
                    "end_at_step": ("INT", {"default": 10000, "min": 0, "max": 10000}),
                    "return_with_leftover_noise": ("BOOLEAN", {"default": False}),
                    "preview_method": (["auto", "latent2rgb", "taesd", "vae_decoded_only", "none"],),
                    "vae_decode": (["true", "true (tiled)", "false"],),
                    },
                    "optional": {
                        "optional_vae": ("VAE",),
                    },
                    "hidden": {
                        "prompt": "PROMPT", 
                        "extra_pnginfo": "EXTRA_PNGINFO", 
                        "my_unique_id": "UNIQUE_ID",
                    },
                }

    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING", "SAMPLER", "SIGMAS", "LATENT","LATENT", "IMAGE", "VAE",)
    RETURN_NAMES = ("model", "positive", "negative", "sampler", "sigmas", "output", "denoised_output", "image", "vae", )
    FUNCTION = "sample"
    CATEGORY = "More Efficient Samplers"

    def sample(self, model, add_noise, noise_seed, cfg, positive, negative, sampler, sigmas, latent_image, start_at_step, end_at_step, return_with_leftover_noise, preview_method, vae_decode, optional_vae=(None,), prompt=None, extra_pnginfo=None, my_unique_id=None):
        latent = latent_image
        latent_image = latent["samples"]
        latent = latent.copy()
        latent_image = comfy.sample.fix_empty_latent_channels(model, latent_image)
        latent["samples"] = latent_image

        # Rename the vae variable
        vae = optional_vae
        # If vae is not connected, disable vae decoding
        if vae == (None,) and vae_decode != "false":
            print(f"{warning('Sampler Custom Ultra Advanced Warning:')} No vae input detected, proceeding as if vae_decode was false.\n")
            vae_decode = "false"
        
        # ------------------------------------------------------------------------------------------------------
        def vae_decode_latent(vae, out, vae_decode):
            return VAEDecodeTiled().decode(vae,out,320)[0] if "tiled" in vae_decode else VAEDecode().decode(vae,out)[0]
        # ---------------------------------------------------------------------------------------------------------------

        noise_mask = None
        if "noise_mask" in latent:
            noise_mask = latent["noise_mask"]
        
        def process_latents():
            x0_output = {}
            # Initialize output variables
            out = out_denoised = images = preview = previous_preview_method = None

            if not add_noise:
                noise = Noise_EmptyNoise().generate_noise(latent)
            else:
                noise = Noise_RandomNoise(noise_seed).generate_noise(latent)
        
            try:
                # Change the global preview method (temporarily)
                set_preview_method(preview_method)
                
                x0_output = {}
                callback = latent_preview.prepare_callback(model, sigmas.shape[-1] - 1, x0_output)

                disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
        
                disable_noise = False
                if not add_noise:
                    disable_noise = True
        
                if disable_noise:
                    noise = torch.zeros(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout, device="cpu")
                else:
                    batch_inds = latent["batch_index"] if "batch_index" in latent else None
                    noise = comfy.sample.prepare_noise(latent_image, noise_seed, batch_inds)
    
                force_full_denoise = True
                if return_with_leftover_noise:
                    force_full_denoise = False
        
                device = comfy.model_management.intermediate_device()
                model_options = model.model_options
                start_step = start_at_step
                last_step = end_at_step
                denoise_mask = noise_mask

                samples = sample_custom_ultra(model, device, noise, sampler, positive, negative, cfg, model_options, latent_image, start_step, last_step, force_full_denoise, denoise_mask, sigmas, callback, disable_pbar, noise_seed)
                samples = samples.to(comfy.model_management.intermediate_device())
                
                out = latent.copy()
                out["samples"] = samples
                if "x0" in x0_output:
                    out_denoised = latent.copy()
                    out_denoised["samples"] = model.model.process_latent_out(x0_output["x0"].cpu())
                else:
                    out_denoised = out

                previous_preview_method = global_preview_method()

                # ---------------------------------------------------------------------------------------------------------------
                # Decode image if not yet decoded
                if "true" in vae_decode:
                    if images is None:
                        images = vae_decode_latent(vae, out, vae_decode)
                        # Store decoded image as base image of no script is detected
                        store_ksampler_results("image", my_unique_id, images)

                # Define preview images
                if preview_method == "none" or (preview_method == "vae_decoded_only" and vae_decode == "false"):
                    preview = {"images": list()}
                elif images is not None:
                    preview = PreviewImage().save_images(images, prompt=prompt, extra_pnginfo=extra_pnginfo)["ui"]

                # Define a dummy output image
                if images is None and vae_decode == "false":
                    images = SamplerCustomUltraAdvancedEfficient.empty_image

            finally:
                # Restore global changes
                set_preview_method(previous_preview_method)
              
            return out, out_denoised, preview, images
        
        # ---------------------------------------------------------------------------------------------------------------
        # Clean globally stored objects of non-existant nodes
        globals_cleanup(prompt)
        # ---------------------------------------------------------------------------------------------------------------
        out, out_denoised, preview, images = process_latents()

        result = (model, positive, negative, sampler, sigmas, 
                  out, out_denoised, images, vae,)

        if preview is None:
            return {"result": result}
        else:
            return {"ui": preview, "result": result}
