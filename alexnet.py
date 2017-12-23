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
		return F.softmax(x,dim=-1)

def alexnet(pretrained=False, **kwargs):
	r"""AlexNet model architecture from the
	`"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
	Args:
		pretrained (bool): If True, returns a model pre-trained on ImageNet
	"""
	model = AlexNet(**kwargs)
	if pretrained:
		model.load_state_dict(model_zoo.load_url(model_urls['alexnet']))
	return model


# Refer to this: http://www.image-net.org/challenges/LSVRC/2012/supervision.pdf
if __name__ == "__main__":
	# sample forward
	is_cuda = torch.cuda.is_available()    
	net = alexnet(True)
	
	# We don't change the already learnt parameters as of now
	for param in net.parameters():
		param.requires_grad = False

	finetune = MyNet()
	# inp = Variable(torch.rand(5, 3, 224, 224))

	if is_cuda:
		finetune = finetune.cuda()
		net = net.cuda()
		# inp = inp.cuda()

	# y = net(inp)
	# y = finetune(y)
	# print(y.shape)
	
	# Data augmentation and normalization for training

	data_transforms = {
		# Training data transforms
		'train': transforms.Compose([
			transforms.RandomResizedCrop(224),
			# transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
		]),

		# Validation Data Transforms, to be needed later for finetuning
		'val': transforms.Compose([
			transforms.Resize(256),
			transforms.CenterCrop(224),
			transforms.ToTensor(),
			transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
		]),
	}

	# Data Loading
	# The directory containing the train and the val folders
	data_dir = '../'
	# For now only the train images are being used , can extend the list to include for 'val' later

	# Image datasets with the transforms applied
	image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),data_transforms[x])
						for x in ['train']}

	# Dataloaders to load data in batches from the datasets
	dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,shuffle=True, num_workers=4)
						for x in ['train']}
	# Dataset Sizes
	dataset_sizes = {x: len(image_datasets[x]) 
						for x in ['train']}
	# Class Names
	class_names = image_datasets['train'].classes

	# Dataloder for the training set
	dataloader = dataloaders['train']

	# Training Code Incomplete as of now
	# Cross Entropy Loss
	criterion = nn.CrossEntropyLoss()
	# Stochastic Gradient Descent
	optimizer = optim.SGD(finetune.parameters(), lr=0.001, momentum=0.9)
	
	running_loss = 0.0
	for epoch in range(1):
		for i, data in enumerate(dataloader, 0):
			# get the inputs
			inputs, labels = data

			# wrap them in Variable
			inputs, labels = Variable(inputs), Variable(labels)

			# zero the parameter gradients
			optimizer.zero_grad()

			# forward + backward + optimize
			outputs = finetune(net(inputs))
			loss = criterion(outputs, labels)
			loss.backward()
			optimizer.step()

			running_loss += loss
			if i>0 and i%50 == 0:
				print(i,running_loss/50)
				running_loss = 0.0


	# Defining an iterator for the dataloader for train
	dataiter = iter(dataloaders['train'])

	# Taking a batch of image dataset which are already in tensor form and ready to be fed into the network
	images, labels = dataiter.next()

	# print(images)
	print(labels)

	outputs = finetune(net(Variable(images)))
	print(outputs)

	val,ind = torch.max(outputs,1)
	print(val,ind)