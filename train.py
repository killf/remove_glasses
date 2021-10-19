import torch
from torchvision.transforms import *

from datasets import UnpairedImagesFolder
from solver import Solver

transform = Compose([Resize(128), ToTensor()])
train_data = UnpairedImagesFolder("/data/face/parsing/dataset/wanda_glasses/data3", transform=transform)

Solver(epochs=100).fit(train_data=train_data, train_batch_size=32)
