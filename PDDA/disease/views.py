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

def dummy(image_arr, crop_name):
	print(image_arr)
	return {'foo':crop_name}