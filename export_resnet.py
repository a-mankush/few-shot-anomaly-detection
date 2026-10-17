# PROTOTYPE: one-time export of WRN50-2 (PatchCore-style layer2+layer3 patch features) to ONNX. uv run export_resnet.py [res...]
import sys, torch, torch.nn.functional as F, torchvision as tv
class W(torch.nn.Module):
    def __init__(s):
        super().__init__(); m = tv.models.wide_resnet50_2(weights="IMAGENET1K_V1").eval()
        s.stem = torch.nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool, m.layer1); s.l2, s.l3 = m.layer2, m.layer3
    def forward(s, x):
        f2 = s.l2(s.stem(x)); f3 = s.l3(f2)
        f2, f3 = F.avg_pool2d(f2, 3, 1, 1), F.avg_pool2d(f3, 3, 1, 1)   # PatchCore local aggregation
        f3 = F.interpolate(f3, size=f2.shape[-2:], mode="bilinear", align_corners=False)
        f = torch.cat([f2, f3], 1).flatten(2).transpose(1, 2)           # (1, h*w, 1536)
        return F.avg_pool1d(f, 2)                                        # (1, h*w, 768): channel-pair mean (adaptive 1024 does not export)
for r in map(int, sys.argv[1:] or [224, 448]):
    torch.onnx.export(W(), torch.zeros(1, 3, r, r), f"models/wrn50_{r}.onnx", input_names=["x"], output_names=["last"], opset_version=17, dynamo=False)
    print("exported", r)
