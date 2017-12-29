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


# Loading some random model only for functionality testing
net = alexnet(True)
finetune = finenet(True)

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
			transforms.RandomCrop(256, 256),
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
def upload_image_and_get_results(request):
	if request.method == 'POST':
		crop_name = request.POST['crop_name'] 
		crop_image = Image.open(request.FILES['crop_image'])
		output_dict = dummy(crop_image, crop_name)
		return JsonResponse(output_dict, safe=False)
	else:
		return HttpResponse("<h2>Wrong Page</h2>")

#Change the dummy function to our specific function
def dummy(image_arr, crop_name):
	
	input_image = (data_transforms['val'](image_arr)).cuda()
	outputs = finetune(net(Variable(input_image.unsqueeze(0))))

	# Converting the numerical outputs to probabilities
	soft = nn.Softmax()
	prob = soft(outputs)

	# Number of results
	num_predictions = 5
	val,ind = torch.topk(prob, num_predictions)
	print val, ind, prob.sum()

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