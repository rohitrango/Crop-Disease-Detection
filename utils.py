# Random utilities
import os
import json
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from skimage import io
import torch
import numpy as np

# Divides the images into training/testing files
def divide_training_testing(training_ratio=.8, image_dir="images"):
	for r, dirs, files in os.walk(image_dir):
		if len(files) == 0:
			for _dir in dirs:
				os.makedirs(os.path.join("train", _dir))
				os.makedirs(os.path.join("test", _dir))
		else:
			m = int(training_ratio*len(files))
			training_files = files[:m]
			testing_files  = files[m:]
			for filename in training_files:
				with open(os.path.join(r, filename), 'rb') as img:
					with open(os.path.join(r.replace(image_dir, "train"), filename), 'wb') as wr:
						wr.write(img.read())

			for filename in testing_files:
				with open(os.path.join(r, filename), 'rb') as img:
					with open(os.path.join(r.replace(image_dir, "test"), filename), 'wb') as wr:
						wr.write(img.read())


def generate_labels_mapping(filename="labels.json"):
	with open(filename) as f:
		labels = json.loads(f.read())
	labels_to_num = { v: i for i, v in enumerate(labels)}
	num_to_labels = { i: v for i, v in enumerate(labels)}
	return labels_to_num, num_to_labels

# Class for custom dataset
class PlantVillageDataset(Dataset):

	def __init__(self, image_dir, is_training=True):
		self.image_dir = image_dir
		self.is_training = is_training
		self.image_list = []
		
		# generate label mapping
		l, r = generate_labels_mapping()
		self.labels_to_num = l
		self.num_to_labels = r

		# appropriate transform based on training or testing
		if is_training:
			self.transform = transforms.Compose([
					transforms.ToPILImage(),
					transforms.Scale(256, 256),
					transforms.RandomCrop(224),
					transforms.ToTensor(),
					transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
				])
		else:
			self.transform = transforms.Compose([
						transforms.ToPILImage(),
						transforms.Scale(256, 256),
						transforms.CenterCrop(224),
						transforms.ToTensor(),
						transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
				])

		# generate all paths
		for r, dirs, files in os.walk(image_dir):
			for file in files:
				self.image_list.append(os.path.join(r, file))

		# generate plant label mapping
		self.plants_list = map(lambda x: x.split("/")[1].split("___")[0], self.image_list)
		self.plants_list = sorted(list(set(self.plants_list)))

		self.plant_labels_to_num = {v: i for i, v in enumerate(self.plants_list)}
		self.num_to_plant_labels = {i: v for i, v in enumerate(self.plants_list)}

	def __len__(self):
		return len(self.image_list)

	def __getitem__(self, idx):
		img_path = self.image_list[idx]
		img = io.imread(img_path)
		img = self.transform(img)

		label = torch.from_numpy(
			np.array([ self.labels_to_num[img_path.split("/")[1]] ] ))

		plant_label = torch.from_numpy(np.array([ self.plant_labels_to_num[img_path.split("/")[1].split("___")[0]] ] ))

		return img, plant_label, label
