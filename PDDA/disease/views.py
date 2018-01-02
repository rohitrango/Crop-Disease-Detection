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
from django.conf import settings
from .torchmodels import *
# Create your views here.
# Load the labels one-time
with open(os.path.join(settings.BASE_DIR, "labels.json")) as f:
	idx_to_labels = json.loads(f.read())

with open(os.path.join(settings.BASE_DIR, "labels_v2.json")) as f:
	idx_to_labels_v2 = json.loads(f.read())

with open(os.path.join(settings.BASE_DIR, "plant_labels.json")) as f:
	plant_labels = json.loads(f.read())

# Loading some random model only for functionality testing
net = alexnet(True)
net2 = alexnet()
finetune = finenet(True)
finetune2 = MyNet2()

net2.load_state_dict(torch.load(os.path.join(settings.BASE_DIR, "torchmodels/alexnet_v2.pt")))
finetune2.load_state_dict(torch.load(os.path.join(settings.BASE_DIR, "torchmodels/finetune_v2.pt")))
# Load the pretrained model
# net = alexnet(True)
# finetune = finenet(True)

is_cuda = torch.cuda.is_available()	
if is_cuda:
	finetune = finetune.cuda()
	net = net.cuda()
	finetune2 = finetune2.cuda()
	net2 = net2.cuda()

# Data transformation for forward pass
data_transforms = {

	'val': transforms.Compose([
			transforms.CenterCrop(224),
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406],
								 std=[0.229, 0.224, 0.225])
	])
}

@csrf_exempt
def home(request):
	if request.method == 'GET':
		return render(request, 'home.html', {})
	else:
		return HttpResponseRedirect(reverse("success"))

def success(request):
	return HttpResponse("<h2>Image Upload successful</h2>")

# Expect a POST Request with the MultiPart Data with the name of crop_image and a text data with the name of crop_name
@csrf_exempt
def upload_image_and_get_results(request):
	if request.method == 'POST':
		crop_name = request.POST['crop_name'] 
		crop_image = Image.open(request.FILES['crop_image'])
		if plant_labels.has_key(crop_name):
			crop_index = plant_labels[crop_name]
			output_dict = predict_with_name(crop_image, crop_index)
		else:
			output_dict = predict_without_name(crop_image)
		return JsonResponse({
			'status' : True,
			'response': output_dict,
			}, safe=False)
	else:
		return JsonResponse({
			'status' : False,
			}, safe=False)


def predict_without_name(image_arr):
	print("Without name \n\n\n")
	input_image = (data_transforms['val'](image_arr)).cuda()
	outputs = finetune(net(Variable(input_image.unsqueeze(0))))

	# Converting the numerical outputs to probabilities
	soft = nn.Softmax()
	prob = soft(outputs)

	# Number of results
	num_predictions = 5
	val,ind = torch.topk(prob, num_predictions)
	# print val, ind, prob.sum()

	# Fetch from json
	final = []
	for i in xrange(num_predictions):
		final.append({
			'rank'			:	i,
			'category'		: 	ind[0].data[i],
			'category_name'	:  	idx_to_labels[str(ind[0].data[i])],
			'prob'			:	val[0].data[i],
		})

	return final

def predict_with_name(image_arr, crop_idx):
	print("With name \n\n\n")
	print("Crop index: ", crop_idx)

	plant_label = Variable(torch.LongTensor(np.array([crop_idx])).cuda())
	input_image = (data_transforms['val'](image_arr)).cuda()
	o1 = net2(Variable(input_image.unsqueeze(0)))
	outputs = finetune2(o1, plant_label)

	# Converting the numerical outputs to probabilities
	soft = nn.Softmax()
	prob = soft(outputs)

	# Number of results
	num_predictions = 5
	val,ind = torch.topk(prob, num_predictions)
	# print val, ind, prob.sum()

	# Fetch from json
	final = []
	for i in xrange(num_predictions):
		final.append({
			'rank'			:	i,
			'category'		: 	ind[0].data[i],
			'category_name'	:  	idx_to_labels[str(ind[0].data[i])],
			'prob'			:	val[0].data[i],
		})

	return final
