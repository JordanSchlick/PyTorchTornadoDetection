import os
file_directory=os.path.dirname(__file__)
if file_directory:
    os.chdir(file_directory)

import math
import time
import sys
import multiprocessing
from matplotlib import pyplot as plt
import numpy as np
import pandas
import torch
import torchvision.utils
import torch.utils.tensorboard
import torch.utils.data
import model
import dataset


device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
#device_str = "cpu"
device = torch.device(device_str)

dataset_files = dataset.DirectoryTrainTest("./data/Radar/l2data", train_percentage=90)
print("Data split", len(dataset_files.train_list), len(dataset_files.test_list))
thread_count = max(math.ceil(multiprocessing.cpu_count() / 2) - 2, 1)
tornado_dataset = dataset.TornadoDataset(dataset_files.test_list, thread_count=thread_count, buffer_size=thread_count * 2, section_size=256, auto_shuffle=False, cache_results=True)
custom_data_loader = dataset.CustomTorchLoader(tornado_dataset, batch_size=16, device=device)


tornado_detection_model = model.TornadoDetectionModel()

pytorch_total_params = sum(p.numel() for p in tornado_detection_model.parameters())
print("parameter count", pytorch_total_params)

if True:
	# distribute across gpus
	# if device_str == "cpu" or os.name == 'nt':
	# 	torch.distributed.init_process_group(backend='gloo')
	# else:
	# 	torch.distributed.init_process_group(backend='nccl')
	# tornado_detection_model = torch.nn.parallel.DistributedDataParallel(tornado_detection_model)
	tornado_detection_model = torch.nn.DataParallel(tornado_detection_model)
	
tornado_detection_model.to(device)
tornado_detection_model.train(False)

loss_function = model.MaskLoss()
loss_function.to(device)

model_file = "saved_model.pt"
if os.path.isfile(model_file):
	saved_data = torch.load(model_file)
	tornado_detection_model.load_state_dict(saved_data["state_dict_model"])
	step = saved_data["step"]
	del saved_data
	print("loaded model from step", step)

arg = ""
try:
	arg = sys.argv[1]
except:
	pass
if arg == "csv":
	# create a csv data containing info about the model
	tornados = []

	# run through testing set and collect data
	beginning = True
	while tornado_dataset.location > 100 + thread_count * 2 or beginning:
	# for i in range(100):
		if tornado_dataset.location > 100 + thread_count * 2:
			beginning = False
		batch = custom_data_loader.next()
		print("got batch with ids ", batch["location_in_dataset"])
		
		input_data = batch["data"]
		mask = batch["mask"]
		
		with torch.cuda.amp.autocast():
			output = tornado_detection_model(input_data)
			loss, extra_loss_info = loss_function(output, mask)
		
		loss = loss.item()
		print("loss", loss)
		outside_mask = extra_loss_info["outside_mask"].detach().cpu().numpy()
		inside_mask = extra_loss_info["inside_mask"].detach().cpu().numpy()
		for i in range(len(batch["tornado_info"])):
			tornado_info = batch["tornado_info"][i]
			true_positive_loss = inside_mask[i]
			false_positive_loss = outside_mask[i]
			tornado_info["true_positive_loss"] = true_positive_loss
			tornado_info["false_positive_loss"] = false_positive_loss
			narrative = tornado_info["narrative"]
			del tornado_info["narrative"]
			tornado_info["narrative"] = narrative
			tornados.append(tornado_info)
		
	df = pandas.DataFrame(tornados)
	df.to_csv("evaluation_data.csv")

elif arg == "image":
	image = 0
	os.makedirs("evaluation_images", exist_ok=True)
	# display output images
	def matplotlib_imshow(img, one_channel=False):
		global image
		if one_channel:
			img = img.mean(dim=0)
		#img = img / 2 + 0.5     # unnormalize
		fig = plt.figure(figsize=(10,10))
		ax = fig.add_subplot(111)
		npimg = img.cpu().numpy()
		if one_channel:
			ax.imshow(npimg, cmap="Greys")
		else:
			ax.imshow(np.transpose(npimg, (1, 2, 0)))
		#plt.show()
		plt.savefig("evaluation_images/"+str(image)+".png")
		image += 1
	beginning = True
	while tornado_dataset.location > 100 + thread_count * 2 or beginning:
	# for i in range(100):
		if tornado_dataset.location > 100 + thread_count * 2:
			beginning = False
		batch = custom_data_loader.next()
		print("got batch with ids ", batch["location_in_dataset"])
		input_data = batch["data"]
		mask = batch["mask"]
		
		with torch.cuda.amp.autocast():
			output = tornado_detection_model(input_data)
			loss, extra_loss_info = loss_function(output, mask)
		
		images = torch.stack([
			mask.cpu()[:16],
			# torch.maximum(input_data.cpu()[:16,0,0,:,:] - 0.5, torch.tensor(0)),
			output.cpu()[:16],
			input_data.cpu()[:16,0,0,:,:],
		], 1)
		images.clamp_(0, 1)
		#images = torch.nn.functional.max_pool2d(images, 4)
		img_grid = torchvision.utils.make_grid(images, nrow=4)
		#matplotlib_imshow(img_grid)
		torchvision.utils.save_image(img_grid, "evaluation_images/"+str(image)+".png")
		image += 1
elif arg == "stats":
	items = 0
	outside_correct = 0
	inside_correct = 0
	items_close = 0
	outside_correct_close = 0
	inside_correct_close = 0
	items_closer = 0
	outside_correct_closer = 0
	inside_correct_closer = 0
	items_range = 0
	outside_correct_range = 0
	inside_correct_range = 0
	beginning = True
	while tornado_dataset.location > 100 + thread_count * 2 or beginning:
	# for i in range(100):
		if tornado_dataset.location > 100 + thread_count * 2:
			beginning = False
		batch = custom_data_loader.next()
		print("got batch with ids ", batch["location_in_dataset"])
		
		input_data = batch["data"]
		mask = batch["mask"]
		
		with torch.cuda.amp.autocast():
			output = tornado_detection_model(input_data)
			loss, extra_loss_info = loss_function(output, mask)
		
		loss = loss.item()
		print("loss", loss)
		outside_mask = extra_loss_info["outside_mask"].detach().cpu().numpy()
		inside_mask = extra_loss_info["inside_mask"].detach().cpu().numpy()
		items += len(inside_mask)
		for i in range(len(inside_mask)):
			# values are offset due to the mean part of the loss function
			if inside_mask[i] < 0.55:
				inside_correct += 1
			if outside_mask[i] < 0.45:
				outside_correct += 1
			tornado_info = batch["tornado_info"][i]
			if tornado_info["radar_distance"] < 400000:
				items_range += 1
				if inside_mask[i] < 0.55:
					inside_correct_range += 1
				if outside_mask[i] < 0.45:
					outside_correct_range += 1
			if tornado_info["radar_distance"] < 200000:
				items_close += 1
				if inside_mask[i] < 0.55:
					inside_correct_close += 1
				if outside_mask[i] < 0.45:
					outside_correct_close += 1
			if tornado_info["radar_distance"] < 100000:
				items_closer += 1
				if inside_mask[i] < 0.55:
					inside_correct_closer += 1
				if outside_mask[i] < 0.45:
					outside_correct_closer += 1
				
				
	
	print("total", items)
	print("inside correct (true positives)", inside_correct, str(round(inside_correct / items * 100)) + "%")
	print("outside correct (true negatives)", outside_correct, str(round(outside_correct / items * 100)) + "%")
	
	print("total less than 400 km", items_range)
	print("inside correct less than 400 km (true positives)", inside_correct_range, str(round(inside_correct_range / items_range * 100)) + "%")
	print("outside correct less than 400 km (true negatives)", outside_correct_range, str(round(outside_correct_range / items_range * 100)) + "%")
	
	print("total less than 200 km", items_close)
	print("inside correct less than 200 km (true positives)", inside_correct_close, str(round(inside_correct_close / items_close * 100)) + "%")
	print("outside correct less than 200 km (true negatives)", outside_correct_close, str(round(outside_correct_close / items_close * 100)) + "%")
	
	print("total less than 100 km", items_closer)
	print("inside correct less than 100 km (true positives)", inside_correct_closer, str(round(inside_correct_closer / items_closer * 100)) + "%")
	print("outside correct less than 100 km (true negatives)", outside_correct_closer, str(round(outside_correct_closer / items_closer * 100)) + "%")
	
		
else:
	print("usage python evaluate.py [csv|image|stats]")