import sys
import torch, torchvision, transformers, peft, accelerate, datasets, cv2, ultralytics, fastapi
print("python", sys.version.split()[0])
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("torchvision", torchvision.__version__)
print("transformers", transformers.__version__)
print("peft", peft.__version__)
print("accelerate", accelerate.__version__)
print("datasets", datasets.__version__)
print("ultralytics", ultralytics.__version__)
print("opencv", cv2.__version__)
print("fastapi", fastapi.__version__)
print("cuda_available", torch.cuda.is_available(), "device_count", torch.cuda.device_count())
if torch.cuda.is_available():
    x = torch.ones(8, 8, device="cuda:0")
    print("cuda_matmul_ok", (x @ x).sum().item())
