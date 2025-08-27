import torch

# Load MiDaS small from Intel repo
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
tf = transforms.small_transform
midas.eval()

# Dummy input [1,3,192,320]
dummy = torch.rand(1, 3, 192, 320)

torch.onnx.export(
    midas,
    dummy,
    "midas_small_192x320.onnx",
    input_names=["camera_input"],
    output_names=["inv_depth"],
    opset_version=12,
    do_constant_folding=True,
    dynamic_axes=None
)
print("Exported: midas_small_192x320.onnx")
