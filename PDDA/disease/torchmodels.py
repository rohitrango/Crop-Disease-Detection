import torch.utils.model_zoo as model_zoo
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import torch.nn.functional as F
import torch.optim as optim
import json
from django.conf import settings

class AlexNet(nn.Module):

	def __init__(self, num_classes=1000):
		super(AlexNet, self).__init__()
		self.features = nn.Sequential(
			nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=3, stride=2),
			nn.Conv2d(64, 192, kernel_size=5, padding=2),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=3, stride=2),
			nn.Conv2d(192, 384, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(384, 256, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(256, 256, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=3, stride=2),
		)
		
		self.classifier = nn.Sequential(
			nn.Dropout(),
			nn.Linear(256 * 6 * 6, 4096),
			nn.ReLU(inplace=True),
			nn.Dropout(),
			nn.Linear(4096, 4096),
			nn.ReLU(inplace=True),
			nn.Linear(4096, num_classes),
		)

	def forward(self, x):
		x = self.features(x)
		x = x.view(-1, 256 * 6 * 6)
		x = self.classifier(x)
		return x

class MyNet(nn.Module):

	def __init__(self, num_classes=38):
		super(MyNet, self).__init__()
		self.fc1 = nn.Sequential(
			nn.ReLU(inplace=True),
			nn.Linear(1000, 38),
		)

	def forward(self, x):
		x = self.fc1(x)
		return x

def alexnet(pretrained=False, **kwargs):
	r"""
	Args:
		pretrained (bool): If True, returns a trained model
	"""
	model = AlexNet(**kwargs)
	if pretrained:
		model.load_state_dict(torch.load(os.path.join(settings.BASE_DIR, "torchmodels/alexnet.pt")))
	return model

def finenet(pretrained=False, **kwargs):
	r"""Args:
		pretrained (bool): If True, returns a trained model
	"""
	model = MyNet(**kwargs)
	if pretrained:
		model.load_state_dict(torch.load(os.path.join(settings.BASE_DIR, "torchmodels/finetune.pt")))
	return model