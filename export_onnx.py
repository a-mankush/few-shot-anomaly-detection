# PROTOTYPE: one-time export of DINOv2 ViT-S/14 to ONNX (needs torch). Outputs [last, mid] patch tokens.
import sys, torch
class M(torch.nn.Module):
    def __init__(s, m): super().__init__(); s.m = m
    def forward(s, x):
        a, b = s.m.get_intermediate_layers(x, n=[8, 11], norm=True)  # 0-based: layer 9 (mid) and 12 (last)
        return b, a
m = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").eval()
for r in map(int, sys.argv[1:] or [224, 448]):
    torch.onnx.export(M(m), torch.zeros(1, 3, r, r), f"models/dinov2_s14_{r}.onnx", input_names=["x"],
                      output_names=["last", "mid"], opset_version=17, dynamo=False)
    print("exported", r)
