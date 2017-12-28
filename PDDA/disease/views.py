from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.http import HttpResponseRedirect
from django.http import HttpResponse
from django.urls import reverse
from django.template import loader
from django.core.files.images import ImageFile
from django.core.files import File
from PIL import Image
import numpy as np
from django.http import JsonResponse

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


# Create your views here.

@csrf_exempt
def home(request):
	if request.method == 'GET':
		return render(request, 'home.html', {})
	else:
		return HttpResponseRedirect(reverse("success"))

def success(request):
	return HttpResponse("<h2>Image Upload successful</h2>")

# Expect a POST Request with the MultiPart Data with the name of crop_image and a text data with the name of crop_name
def upload_image_and_get_results(request):
	if request.method == 'POST':
		crop_name = request.POST['crop_name'] 
		crop_image = Image.open(request.FILES['crop_image'])
		crop_image_arr = np.array(crop_image)
		# return HttpResponse("<h2>Image Upload successful</h2>")
		output_dict = dummy(crop_image_arr, crop_name)
		return JsonResponse(output_dict)
	else:
		return HttpResponse("<h2>Wrong Page</h2>")

#Change the dummy function to our specific function
def dummy(image_arr, crop_name):
	
	# print(image_arr)

	image_tensor = torch.from_numpy(image_arr)
	# print image_tensor
	# print type(image_tensor)

	# Loading some random model only for functionality testing
	net = alexnet()
	finetune = finenet()

	# Load the pretrained model
	# net = alexnet(True)
	# finetune = finenet(True)

	is_cuda = torch.cuda.is_available()	
	if is_cuda:
		finetune = finetune.cuda()
		net = net.cuda()

	# Data transformation for forward pass
	data_transforms = {

		'val': transforms.Compose([
				transforms.ToPILImage(),
				transforms.Resize(256),
				transforms.CenterCrop(224),
				transforms.ToTensor(),
				transforms.Normalize(mean=[0.485, 0.456, 0.406],
									 std=[0.229, 0.224, 0.225])
		])
	}

	input_image = data_transforms['val'](image_tensor)

	# Unsqueeze to convert to a 4D tensor where the first dimension(dim 0) is added and has value 1 as there is only one image in the batch.

	outputs = finetune(net(Variable(input_image.unsqueeze(0))))
	# print outputs

	# Converting the numerical outputs to probabilities
	soft = nn.Softmax(dim=1)
	prob = soft(outputs)
	# print prob


	# Number of results
	num_predictions = 5

	val,ind = torch.topk(prob,num_predictions)
	print val, ind

	json_output = {}

	json_file = open('/home/samarjeet/Desktop/Microsoft Code Fun Do 2018/CodeFunDo/PDDA/disease/labels.json')

	json_data = json_file.read()
	plant_name = json.loads(json_data)

	for i in range(num_predictions):
		plant_key = 'plant'+str(i+1)
		prob_key = 'prob'+str(i+1)
		json_output[plant_key] = plant_name[ind[0].data[i]]
		json_output[prob_key] = val[0].data[i]

	json_file.close()

	return json_output

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
		model.load_state_dict(torch.load("/home/samarjeet/Desktop/Microsoft Code Fun Do 2018/alexnet.pt"))
	return model

def finenet(pretrained=False, **kwargs):
	r"""Args:
		pretrained (bool): If True, returns a trained model
	"""
	model = MyNet(**kwargs)
	if pretrained:
		model.load_state_dict(torch.load("/home/samarjeet/Desktop/Microsoft Code Fun Do 2018/finetune.pt"))
	return model