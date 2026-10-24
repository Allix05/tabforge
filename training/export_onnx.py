"""Export the guitar model (fine-tuned Basic Pitch + string head) to ONNX for
onnxruntime-web, and check the ONNX output matches PyTorch.

  python export_onnx.py --base guitar6            # -> ../model-guitar/tabforge-guitar.onnx
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from common import AUDIO_N_SAMPLES, REPO, load_model
from string_head import TabModel


class Exported(torch.nn.Module):
    def __init__(self, tab):
        super().__init__()
        self.tab = tab

    def forward(self, audio):
        o = self.tab(audio)
        return o["note"], o["onset"], o["contour"], o["string"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="guitar6")
    ap.add_argument("--head", default="C:/tftrain/runs/strings/head.pth")
    ap.add_argument("--out", default=str(REPO / "model-guitar" / "tabforge-guitar.onnx"))
    ap.add_argument("--untrained-head", action="store_true", help="export test without a trained head")
    args = ap.parse_args()

    base = load_model(f"C:/tftrain/runs/{args.base}/best.pth").cpu()
    tab = TabModel(base)
    if not args.untrained_head:
        tab.head.load_state_dict(torch.load(args.head, map_location="cpu"))
    model = Exported(tab).eval()
    x = torch.randn(2, AUDIO_N_SAMPLES) * 0.1
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model, (x,), str(out), input_names=["audio"], output_names=["note", "onset", "contour", "string"],
        dynamic_axes={"audio": {0: "batch"}, "note": {0: "batch"}, "onset": {0: "batch"}, "contour": {0: "batch"}, "string": {0: "batch"}},
        opset_version=17, dynamo=False,
    )
    import onnxruntime as ort
    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    got = sess.run(None, {"audio": x.numpy()})
    with torch.no_grad():
        ref = model(x)
    for name, a, b in zip(["note", "onset", "contour", "string"], got, ref):
        print(f"{name:8s} shape {a.shape}  max abs diff {np.max(np.abs(a - b.numpy())):.2e}")
    print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
