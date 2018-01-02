from django.conf.urls import url
from .views import *

urlpatterns = [
    url(r'^$', home, name='home'),
    url(r'^success$', success, name='success'),
    url(r'^upload_image_and_get_results/', upload_image_and_get_results, name='upload_image_and_get_results'),
    url(r'^save_entry', save_entry, name='save_entry')
]