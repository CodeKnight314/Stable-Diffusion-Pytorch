# Stable-Diffusion-Pytorch

## Overview
This is an educational repository on fine-tuning publicly available Stale-Diffusion model at HuggingFace. The repository includes a custom Stable-Diffusion Model class configurable for both conditional and unconditional image training. Compared to other impressive Pytorch implementations of Stable-Diffusion, of which there are many, this repository aims to provide an easy framework to work with Stable-Diffusion model on fine-tuning tasks rather than implement from scratch, out of consideration for computational resources and redundancy. 

The curren repository is available to fine-tune on conditional or unconditional fine-tuning. For future developments, this repository will aim to expand the fine-tuning framework to Image-to-Image translation, Image inpainting, and other areas within reason. Results are, or will be, shown below along with instructions on usage and deploymnet.

## Visual Results
<div style="display: flex; justify-content: center; flex-wrap: wrap; gap: 20px; max-width: 90%; margin: 0 auto;">
    <div style="display: flex; justify-content: center; flex-basis: 100%; flex-wrap: wrap;">
        <img src="resources/result_0.png" alt="Stable-Diffusion Image 0" style="width: 30%; margin: 10px;">
        <img src="resources/result_1.png" alt="Stable-Diffusion Image 1" style="width: 30%; margin: 10px;">
        <img src="resources/result_2.png" alt="Stable-Diffusion Image 2" style="width: 30%; margin: 10px;">
    </div>
    <div style="display: flex; justify-content: center; flex-basis: 100%; flex-wrap: wrap;">
        <img src="resources/result_3.png" alt="Stable-Diffusion Image 3" style="width: 30%; margin: 10px;">
        <img src="resources/result_4.png" alt="Stable-Diffusion Image 4" style="width: 30%; margin: 10px;">
        <img src="resources/result_5.png" alt="Stable-Diffusion Image 5" style="width: 30%; margin: 10px;">
    </div>
</div>