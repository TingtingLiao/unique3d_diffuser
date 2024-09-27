import torch 
import numpy as np 
from PIL import Image 
from pipeline import Unique3dDiffusionPipeline
from diffusers import EulerAncestralDiscreteScheduler


pipe = Unique3dDiffusionPipeline.from_pretrained(  
    "Luffuly/unique3d-normal-diffuser", 
    torch_dtype=torch.bfloat16, 
    trust_remote_code=True,  
).to("cuda")
# pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

seed = -1    
generator = torch.Generator(device='cuda').manual_seed(-1)

image = Image.open('/media/mbzuai/Tingting/unique3d-diffuser/hao.png').convert("RGB")

forward_args = dict(
    width=512,
    height=512, 
    width_cond=512,
    height_cond=512, 
    generator=generator,
    guidance_scale=1.5,   
    num_inference_steps=30, 
    num_images_per_prompt=1, 
)  
out = pipe(image, **forward_args).images
# rgb_np = np.hstack([np.array(img) for img in out])
# Image.fromarray(rgb_np).save(f"test.png")
out[0].save(f"out.png")