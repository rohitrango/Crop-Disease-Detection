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
		self.fc1 = nn.Linear(1000, 38)

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
		model.load_state_dict(model_zoo.load_url(model_urls['alexnet']))
	return model

# Refer to this: http://www.image-net.org/challenges/LSVRC/2012/supervision.pdf
if __name__ == "__main__":
	# sample forward
	is_cuda = torch.cuda.is_available()    
	net = alexnet(True)
	finetune = MyNet()
	inp = Variable(torch.rand(5, 3, 224, 224))
	if is_cuda:
		finetune = finetune.cuda()
		net = net.cuda()
		inp = inp.cuda()

	# y = net(inp)
	# y = finetune(y)
	# print(y.shape)

	# Data augmentation and normalization for training
	# Just normalization for validation
	data_transforms = {
		'train': transforms.Compose([
			transforms.RandomResizedCrop(224),
			# transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
		]),
		'val': transforms.Compose([
			transforms.Resize(256),
			transforms.CenterCrop(224),
			transforms.ToTensor(),
			transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
		]),
	}

	
	# Data Loading
	data_dir = '../'
	image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
											  data_transforms[x])
					  for x in ['train']}
	dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
												 shuffle=True, num_workers=4)
				  for x in ['train']}
	dataset_sizes = {x: len(image_datasets[x]) for x in ['train']}
	class_names = image_datasets['train'].classes

	dataiter = iter(dataloaders['train'])
	
	images, labels = dataiter.next()
	print(images,labels)

	outputs = net(Variable(images))
	print(outputs)
