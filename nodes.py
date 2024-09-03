import comfy.sample
import pprint
from comfy.samplers import KSAMPLER
from comfy.k_diffusion.sampling import deis
from tqdm import trange
# from comfy.k_diffusion.sampling import sample_deis
from functools import partial
import comfy.samplers
import torch 
import importlib


x_flux_module = importlib.import_module("custom_nodes.x-flux-comfyui")
# import sampling submodule from x-flux-comfyui
get_schedule = x_flux_module.sampling.get_schedule


import matplotlib.pyplot as plt
import numpy as np
import torch
import os

# copied from comfy.k_diffusion.sampling to fix error when max_order is 1
@torch.no_grad()
def sample_deis(model, x, sigmas, extra_args=None, callback=None, disable=None, max_order=3, deis_mode='tab'):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    x_next = x
    t_steps = sigmas

    coeff_list = deis.get_deis_coeff_list(t_steps, max_order, deis_mode=deis_mode)

    buffer_model = []
    
    # Lists to store magnitudes for plotting
    x_cur_minus_denoised_mags = []
    d_cur_mags = []
    step_mags = []

    for i in trange(len(sigmas) - 1, disable=disable):
        t_cur = sigmas[i]
        t_next = sigmas[i + 1]

        x_cur = x_next

        denoised = model(x_cur, t_cur * s_in, **extra_args)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})

        x_cur_minus_denoised = x_cur - denoised
        d_cur = x_cur_minus_denoised / t_cur

        # Calculate and store magnitudes
        x_cur_minus_denoised_mags.append(torch.norm(x_cur_minus_denoised).item())
        d_cur_mags.append(torch.norm(d_cur).item())

        order = min(max_order, i+1)
        if t_next <= 0:
            order = 1

        if order == 1:          # First Euler step.
            step = (t_next - t_cur) * d_cur
        elif order == 2:        # Use one history point.
            coeff_cur, coeff_prev1 = coeff_list[i]
            step = coeff_cur * d_cur + coeff_prev1 * buffer_model[-1]
        elif order == 3:        # Use two history points.
            coeff_cur, coeff_prev1, coeff_prev2 = coeff_list[i]
            step = coeff_cur * d_cur + coeff_prev1 * buffer_model[-1] + coeff_prev2 * buffer_model[-2]
        elif order == 4:        # Use three history points.
            coeff_cur, coeff_prev1, coeff_prev2, coeff_prev3 = coeff_list[i]
            step = coeff_cur * d_cur + coeff_prev1 * buffer_model[-1] + coeff_prev2 * buffer_model[-2] + coeff_prev3 * buffer_model[-3]

        x_next = x_cur + step

        # Log the magnitude of step
        step_mags.append(torch.norm(step).item())

        if len(buffer_model) == max_order - 1 and max_order > 1:
            for k in range(max_order - 2):
                buffer_model[k] = buffer_model[k+1]
            buffer_model[-1] = d_cur.detach()
        else:
            buffer_model.append(d_cur.detach())

    # Plot the magnitudes
    plt.figure(figsize=(10, 6))
    plt.plot(x_cur_minus_denoised_mags, label='(x_cur - denoised)')
    plt.plot(d_cur_mags, label='d_cur')
    plt.plot(step_mags, label='step')
    plt.xlabel('Step')
    plt.ylabel('Magnitude')
    plt.title('Magnitudes during DEIS sampling')
    plt.legend()
    plt.grid(True)

    # Save the plot
    output_dir = 'outputs'
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f'deis_sampling_magnitudes_order_{max_order}.png'))
    plt.close()

    # Create separate plots for each line
    plt.figure(figsize=(10, 6))
    plt.plot(x_cur_minus_denoised_mags)
    plt.xlabel('Step')
    plt.ylabel('Magnitude')
    plt.title('(x_cur - denoised) Magnitude during DEIS sampling')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f'deis_x_cur_minus_denoised_magnitude_order_{max_order}.png'))
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(d_cur_mags)
    plt.xlabel('Step')
    plt.ylabel('Magnitude')
    plt.title('d_cur Magnitude during DEIS sampling')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f'deis_d_cur_magnitude_order_{max_order}.png'))
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(step_mags)
    plt.xlabel('Step')
    plt.ylabel('Magnitude')
    plt.title('Step Magnitude during DEIS sampling')
    plt.grid(True)

    # Compute AUC
    auc = np.trapz(step_mags)
    print("AUC:", auc)
    
    # Add AUC text to the plot
    plt.text(0.95, 0.95, f'AUC: {auc:.2f}', 
             horizontalalignment='right', 
             verticalalignment='top', 
             transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', alpha=0.5),
             fontsize=22,
             fontweight='bold',
             color='red')

    plt.savefig(os.path.join(output_dir, f'deis_step_magnitude_order_{max_order}.png'))
    plt.close()



    return x_next

class SamplerDEIS:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": 
                    {"order": ("INT", {"default": 1, "min": 1, "max": 4, "step":1, "round": False}),},
                }
    RETURN_TYPES = ("SAMPLER",)
    CATEGORY = "sampling/custom_sampling/samplers"

    FUNCTION = "get_sampler"

    def get_sampler(self, order=1):
        sampling_function = partial(sample_deis, max_order=order)
        sampler = KSAMPLER(sampling_function,)
        return (sampler, )

class FlowMatchScheduler:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {
                     "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                     "width": ("INT", {"default": 1024, "min": 1, "max": 9999}),
                     "height": ("INT", {"default": 1024, "min": 1, "max": 9999}),
                      }
               }
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "sampling/custom_sampling/schedulers"

    FUNCTION = "get_sigmas"

    def get_sigmas(self, steps, width, height):
        sigmas = get_schedule(steps, (width // 8) * (height // 8) // 4, shift=True,)
        if not isinstance(sigmas, torch.Tensor):
            sigmas = torch.tensor(sigmas)
        print(f"Flowmatch sigmas: {sigmas}")
        return (sigmas, )