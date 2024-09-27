# Unique3d-diffuser  

<div align="left">
  <a href='LICENSE'><img src='https://img.shields.io/badge/license-MIT-yellow'></a> 
  <a href='https://wukailu.github.io/Unique3D'><img src='https://img.shields.io/badge/Project-Unique3D-green'></a>
  <a href='https://huggingface.co/Luffuly/unique3d-normal-diffuser'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Normal-blue'></a>
   <a href='https://huggingface.co/Luffuly/unique3d-mvimage-diffuser'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-MVImage-red'></a> 
  <br>
</div>
<br>

A unified diffusers implementation of [Unique3d](https://github.com/AiuniAI/Unique3D).

## Install 
```bash 
pip install torch torchvision torchaudio
pip install -r requirements.txt
```

## Examples 

**Image-to-MVImage** 

```bash 
pipe = Unique3dDiffusionPipeline.from_pretrained( 
    "Luffuly/unique3d-mvimage-diffuser", 
    torch_dtype=torch.float16, 
    trust_remote_code=True,  
    class_labels=torch.tensor(range(4)),
).to("cuda")

seed = -1    
generator = torch.Generator(device='cuda').manual_seed(-1)


image = Image.open('data/boy.png') 
forward_args = dict(
    width=256,
    height=256,
    num_images_per_prompt=4, 
    num_inference_steps=50, 
    width_cond=256,
    height_cond=256, 
    generator=generator,
    guidance_scale=1.5,  
) 

out = pipe(image, **forward_args).images
rgb_np = np.hstack([np.array(img) for img in out])
Image.fromarray(rgb_np).save(f"mv-boy.png")
```

**Image-to-Normal**


## Citation  
```
@misc{wu2024unique3d,
      title={Unique3D: High-Quality and Efficient 3D Mesh Generation from a Single Image}, 
      author={Kailu Wu and Fangfu Liu and Zhihan Cai and Runjie Yan and Hanyang Wang and Yating Hu and Yueqi Duan and Kaisheng Ma},
      year={2024},
      eprint={2405.20343},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```