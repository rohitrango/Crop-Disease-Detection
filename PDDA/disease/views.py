from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.http import HttpResponseRedirect
from django.http import HttpResponse
from django.urls import reverse
from django.template import loader
# Create your views here.

@csrf_exempt
def home(request):
	if request.method == 'GET':
		template = loader.get_template('home.html')
		return HttpResponse(template.render(request))
	else:
		return HttpResponseRedirect(reverse("success"))

def success(request):
	return HttpResponse("<h2>Image Upload successful</h2>")