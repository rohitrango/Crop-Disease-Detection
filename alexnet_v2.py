import torch.utils.model_zoo as model_zoo
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import torch.nn.functional as F
import torch.optim as optim
from utils import *
EPOCHS = 10000

__all__ = ['AlexNet', 'alexnet']


model_urls = {
	'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}

'''
Images should be of size = [batch_size, color, W, H]
'''
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

# We need to finetune this network, and add more layers
# The no of neurons in the final layer is 38, which is the no of crop-disease pairs
class MyNet2(nn.Module):

	def __init__(self, num_classes=38, num_plant_species=14):
		super(MyNet2, self).__init__()

		self.num_plant_species = num_plant_species
		self.num_classes = num_classes

		# first sequential to get the value from alexnet
		self.fc1 = nn.Sequential(
			nn.ReLU(inplace=True),
			nn.Linear(1000, 30),
			nn.ReLU(inplace=True),
		)

		self.embed = nn.Embedding(self.num_plant_species, 20)
		self.fc2 = nn.Sequential(
			nn.Linear(50, self.num_classes)
		)

	def forward(self, x, label):
		x = self.fc1(x)
		embed = self.embed(label)
		x = torch.cat([x, embed], 1)
		x = self.fc2(x)
		return x


def alexnet(pretrained=False, **kwargs):
	r"""AlexNet model architecture from the
	`"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
	Args:
		pretrained (bool): If True, returns a model pre-trained on ImageNet
	"""
	model = AlexNet(**kwargs)
	if pretrained:
		model.load_state_dict(torch.load("PDDA/torchmodels/alexnet_v2.pt"))
	return model


# Refer to this: http://www.image-net.org/challenges/LSVRC/2012/supervision.pdf
if __name__ == "__main__":
	# sample forward
	is_cuda = torch.cuda.is_available()	
	net = alexnet(True)
	# We don't change the already learnt parameters as of now
	# only let the last layer be allowed
	# There are 8 trainable layers, and each layer contains 2 params
	for param in list(net.parameters())[:-2]:
		param.requires_grad = False

	finetune = MyNet2()

	if is_cuda:
		finetune = finetune.cuda()
		net = net.cuda()
	
	finetune.load_state_dict(torch.load("PDDA/torchmodels/finetune_v2.pt"))
	# Data augmentation and normalization for training

	data_transforms = {
		# Training data transforms
		'train': transforms.Compose([
				transforms.RandomSizedCrop(224),
				transforms.RandomHorizontalFlip(),
				transforms.ToTensor(),
				transforms.Normalize(mean=[0.485, 0.456, 0.406],
									 std=[0.229, 0.224, 0.225])
		]),

		# Validation Data Transforms, to be needed later for finetuning
		'val': transforms.Compose([
				transforms.CenterCrop(224),
				transforms.ToTensor(),
				transforms.Normalize(mean=[0.485, 0.456, 0.406],
									 std=[0.229, 0.224, 0.225])
		])
	}

	# Training Code Incomplete as of now
	# Cross Entropy Loss
	criterion = nn.CrossEntropyLoss()
	# Stochastic Gradient Descent

	# https://discuss.pytorch.org/t/how-to-train-several-network-at-the-same-time/4920/6
	parameters = set(finetune.parameters()) | set(net.classifier[-1].parameters())
	optimizer = optim.SGD(parameters, lr=0.005, momentum=0.9)

	# Train dataloaders
	train_dataset = PlantVillageDataset('train')
	dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
	
	losses = []
	running_loss = 0.0
	for epoch in range(EPOCHS):
		for i, data in enumerate(dataloader, 0):
			# get the inputs
			inputs, plant_labels, labels = data

			# wrap them in Variable
			inputs, plant_labels, labels = Variable(inputs).cuda(), Variable(plant_labels.squeeze(1)).cuda(), Variable(labels.squeeze(1)).cuda()

			# zero the parameter gradients
			optimizer.zero_grad()

			# forward + backward + optimize
			o1 = net(inputs)
			outputs = finetune(o1, plant_labels)

			loss = criterion(outputs, labels)
			loss.backward()
			optimizer.step()

			running_loss += loss
			
			# Print the average loss for the last 50 batches
			if i%50 == 0 and i>0:
				print("Epoch: %d, i = %d, Running loss: %f"%(epoch, i, running_loss.data[0]/50))
				losses.append(running_loss.data[0])
				running_loss = 0.0
