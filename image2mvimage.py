import os 
import torch 
import numpy as np 
from PIL import Image  
from pipeline import Unique3dDiffusionPipeline
from utils import load_image, pil_hstack, pil_vstack


if __name__ == "__main__": 
    import argparse 

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default='data/2c4862bbc02651661c2833a651898645_high.png', help="path to input image")
    parser.add_argument("--seed", type=int, default=-1, help="seed for generator")
    parser.add_argument("--refine", type=bool, default=False, help="upsampling image")
    opt = parser.parse_args()
    
    torch.seed()
    
    if opt.image.endswith(".png") or opt.image.endswith(".jpg"):
        images = [opt.image]
    else:
        images = [os.path.join(opt.image, f) for f in os.listdir(opt.image) if f.endswith(".png") or f.endswith(".jpg") or f.endswith(".jpeg")] 
    
    # load pipeline
    pipe = Unique3dDiffusionPipeline.from_pretrained(  
        "Luffuly/unique3d-mvimage-diffuser", 
        torch_dtype=torch.bfloat16, 
        trust_remote_code=True,  
        class_labels = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3]),
    ).to("cuda")

    generator = torch.Generator(device='cuda').manual_seed(opt.seed) 
    forward_args = dict(
        width=256,
        height=256, 
        width_cond=256,
        height_cond=256, 
        guidance_scale=1.5,  
        generator=generator, 
        num_inference_steps=50, 
        num_images_per_prompt=4, 
    )  
    
    os.makedirs("./out/mv-image", exist_ok=True)
    for image in images:  
    
        fname = image.split("/")[-1].split(".")[0]
        image = load_image(image)  
        out = pipe(image, **forward_args).images  
        pil_hstack(out).save(f"./out/mv-image/{fname}.png") 
        # out = [
        #     pil_hstack(pipe(image, **forward_args).images) for _ in range(4) 
        # ]
        # pil_vstack(out).save(f"./out/mv-image/{fname}.png") 
        

