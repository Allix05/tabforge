"""Sanity check: GPU visible, Basic Pitch loads, output shapes are as expected."""

import torch

from common import AUDIO_N_SAMPLES, device, load_model

print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
model = load_model()
with torch.no_grad():
    out = model(torch.randn(2, AUDIO_N_SAMPLES, device=device()) * 0.1)
print({k: tuple(v.shape) for k, v in out.items()})
