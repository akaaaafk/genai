import torch
from model import CNN

model = CNN()

x = torch.randn(4, 3, 64, 64)

output = model(x)

print(output.shape)