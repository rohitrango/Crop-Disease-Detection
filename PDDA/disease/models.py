from __future__ import unicode_literals

from django.db import models

# Create your models here.
class Entry(models.Model):
	deviceID = models.ForeignKey('DeviceID')
	gps_lat = models.FloatField()
	gps_lon = models.FloatField()
	timestamp = models.DateTimeField()
	probability = models.FloatField()
	category_name = models.CharField(max_length=20)

class DeviceID(models.Model):
	deviceID = models.CharField(max_length=60)

class UserPlant(models.Model):
	deviceID = models.ForeignKey("DeviceID")
	plant_name = models.CharField(max_length=30)

