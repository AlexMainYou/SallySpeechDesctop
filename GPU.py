import torch
print(torch.cuda.is_available())  # должно вернуть True, если CUDA доступен
print(torch.version.cuda)
