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
EPOCHS = 2000

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
	r"""AlexNet model architecture from the
	`"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
	Args:
		pretrained (bool): If True, returns a model pre-trained on ImageNet
	"""
	model = AlexNet(**kwargs)
	if pretrained:
		model.load_state_dict(torch.load("PDDA/torchmodels/alexnet.pt"))
	return model


# Refer to this: http://www.image-net.org/challenges/LSVRC/2012/supervision.pdf
if __name__ == "__main__":
	# sample forward
	is_cuda = torch.cuda.is_available()
	pretrained = True
	print("Pretrained network: {0}".format(pretrained))	
	net = alexnet(pretrained).eval()
	# We don't change the already learnt parameters as of now
	# only let the last layer be allowed
	# There are 8 trainable layers, and each layer contains 2 params
	if pretrained:		
		for param in list(net.parameters())[:-2]:
			param.requires_grad = False

	finetune = MyNet().eval()
	finetune.load_state_dict(torch.load("PDDA/torchmodels/finetune.pt"))

	if is_cuda:
		finetune = finetune.cuda()
		net = net.cuda()
	
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
	# criterion = nn.CrossEntropyLoss()
	# Stochastic Gradient Descent

	# https://discuss.pytorch.org/t/how-to-train-several-network-at-the-same-time/4920/6
	if pretrained:		
		parameters = set(finetune.parameters()) | set(net.classifier[-1].parameters())
	else:
		parameters = set(finetune.parameters()) | set(net.parameters())

	# optimizer = optim.SGD(parameters, lr=0.005, momentum=0.9)
	# Train dataloaders
	train_dataset = datasets.ImageFolder(root='test', transform=data_transforms['val'])
	dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=False, num_workers=4)
	
	# Keep track of accuracy
	total_correct = 0.0
	total_count = 0.0

	losses = []

	for i, data in enumerate(dataloader, 0):
		# get the inputs
		inputs, labels = data

		# wrap them in Variable
		inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()
		# zero the parameter gradients
		# optimizer.zero_grad()

		# forward + backward + optimize
		outputs = finetune(net(inputs))
		# loss = criterion(outputs, labels)
		val, ind = torch.max(outputs, 1)
		acc = torch.sum(ind == labels).data[0]

		total_correct += acc
		total_count  += labels.shape[0]
		print("Minibatch acc: {0}/{1}".format(acc, labels.shape[0]))

	print("Accuracy: {}".format(total_correct*1.0/total_count))
