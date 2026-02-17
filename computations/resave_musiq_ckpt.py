import torch

sd = torch.load("/Users/dvuglaf/.cache/torch/hub/pyiqa/musiq_ava_ckpt-e8d3f067.pth", map_location="cpu")

new_sd = {}

for k, v in sd.items():
    if k.startswith("head.0."):
        new_sd[k.replace("head.0.", "head.")] = v
    else:
        new_sd[k] = v

torch.save(new_sd, "musiq_ava_ckpt-e8d3f067.pth")