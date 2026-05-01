import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())

print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))
print("CUDA version (torch):", torch.version.cuda)