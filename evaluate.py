import os
file_directory=os.path.dirname(__file__)
if file_directory:
    os.chdir(file_directory)

import math
import time
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


if os.path.isfile("saved_model.pt"):
	saved_data = torch.load("saved_model.pt")
	tornado_detection_model.load_state_dict(saved_data["state_dict_model"])
	step = saved_data["step"]
	del saved_data
	print("loaded model from step", step)

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