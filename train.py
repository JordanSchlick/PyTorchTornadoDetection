import os
file_directory=os.path.dirname(__file__)
if file_directory:
    os.chdir(file_directory)

import math
import time
import multiprocessing
from matplotlib import pyplot as plt
import numpy as np
import torch
import torchvision.utils
import torch.utils.tensorboard
import torch.utils.data
import model
import dataset

# The farther away the less precise the data becomes
# Limiting it to a closer range like 200 km allows it focus on better data
# set to None to use all data
max_radar_distance = None

device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
#device_str = "cpu"
device = torch.device(device_str)

dataset_files = dataset.DirectoryTrainTest("./data/Radar/l2data", train_percentage=90)
print("Data split", len(dataset_files.train_list), len(dataset_files.test_list))
thread_count = max(math.ceil(multiprocessing.cpu_count() / 2) - 2, 1)
#thread_count = 4
train_file_list = dataset_files.train_list
#train_file_list = list(filter(lambda x: not (".gz" in x), train_file_list))

tornado_dataset = dataset.TornadoDataset(train_file_list, thread_count=thread_count, buffer_size=thread_count * 2, section_size=256, auto_shuffle=True, cache_results=True)
# filter out far away tornados that would be missing data
tornado_dataset = dataset.TornadoDatasetFilter(tornado_dataset, max_radar_distance=max_radar_distance*1000)
custom_data_loader = dataset.CustomTorchLoader(tornado_dataset, batch_size=16, device=device)

tornado_dataset_test = dataset.TornadoDataset(dataset_files.test_list, thread_count=thread_count, buffer_size=thread_count * 2, section_size=256, auto_shuffle=True, cache_results=True)
tornado_dataset_test = dataset.TornadoDatasetFilter(tornado_dataset_test, max_radar_distance=max_radar_distance*1000)
custom_data_loader_test = dataset.CustomTorchLoader(tornado_dataset_test, batch_size=16, device=device)

# torch_tornado_dataset = dataset.TorchDataset(tornado_dataset)
# data_loader = torch.utils.data.DataLoader(torch_tornado_dataset, 16, pin_memory=True, pin_memory_device=device_str) #, num_workers=2, pin_memory=True, pin_memory_device=device_str
# data_iter = iter(data_loader)
#data_iter = iter(torch_tornado_dataset)
# item = next(data_iter)
# item = next(data_iter)
# print(item["data"].shape, item["mask"].shape)
def matplotlib_imshow(img, one_channel=False):
	if one_channel:
		img = img.mean(dim=0)
	#img = img / 2 + 0.5     # unnormalize
	
	npimg = img.cpu().numpy()
	if one_channel:
		plt.imshow(npimg, cmap="Greys")
	else:
		plt.imshow(np.transpose(npimg, (1, 2, 0)))
	plt.show()
# img_grid = torchvision.utils.make_grid(item["data"][:,:3,0], nrow=4)
# matplotlib_imshow(img_grid, one_channel=False)
# img_grid = torchvision.utils.make_grid(item["data"][:,:3,0] / 2 + 0.5, nrow=4)
# matplotlib_imshow(img_grid, one_channel=False)
# tornado_dataset.destroy()

writer = torch.utils.tensorboard.SummaryWriter('runs/tornado')

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

#optimizer = torch.optim.SGD(tornado_detection_model.parameters(), lr=0.001, momentum=0.9)
optimizer = torch.optim.AdamW(tornado_detection_model.parameters(),lr=0.0001)
#optimizer = torch.optim.Adadelta(tornado_detection_model.parameters(),lr=1.0)

loss_function = model.MaskLoss()
loss_function.to(device)

# for step in range(10):
# 	#item = next(data_iter)
	
# 	print("got item", tornado_dataset.next()["file"])
#print("sleeping=================")
#time.sleep(1000000)

# log model to tensorboard
item = custom_data_loader.next()
class TraceModel(torch.nn.Module):
	def __init__(self):
		super().__init__()
		self.tornado_detection_model = model.TornadoDetectionModel()
		self.loss_function = model.MaskLoss()
	def forward(self, input_data, mask):
		output = self.tornado_detection_model(input_data)
		loss, extra_loss_info = self.loss_function(output, mask)
		return loss
trace_model = TraceModel().to("cpu")
writer.add_graph(trace_model, (item["data"].to("cpu"), item["mask"].to("cpu")))
del trace_model
writer.flush()

step = -1


# load model
if os.path.isfile("saved_model.pt"):
	saved_data = torch.load("saved_model.pt")
	tornado_detection_model.load_state_dict(saved_data["state_dict_model"])
	optimizer.load_state_dict(saved_data["state_dict_optimizer"])
	step = saved_data["step"]
	del saved_data
	print("loaded model from step", step)


torch.cuda.empty_cache()
accuracy_total_count = 0
accuracy_inside_count = 0
accuracy_outside_count = 0
for i in range(1000000):
	step += 1
	print("get next =====================")
	# item = next(data_iter)
	item = custom_data_loader.next()
	print("got next =====================")
	input_data = item["data"]
	# input_data = input_data.to(device, non_blocking=True)
	mask = item["mask"]
	# mask = mask.to(device, non_blocking=True)
	
	print(input_data.shape)
	optimizer.zero_grad()
	
	with torch.cuda.amp.autocast():
		output = tornado_detection_model(input_data)
		#loss = torch.mean(torch.maximum(1 - output, torch.tensor(0)))
		loss, extra_loss_info = loss_function(output, mask)
	
	loss.backward()
	optimizer.step()
	
	tornado_detection_model.train(False)
	mean_out = torch.mean(output)
	min_out = torch.min(output)
	max_out = torch.max(output)
	
	loss_value = loss.item()
	outside_mask = extra_loss_info["outside_mask"].detach().cpu().numpy()
	inside_mask = extra_loss_info["inside_mask"].detach().cpu().numpy()
	print("step", step, "loss", loss_value, "mean", mean_out.detach().cpu().numpy(), "min", min_out.detach().cpu().numpy(), "max", max_out.detach().cpu().numpy())
	print("outside_mask", outside_mask)
	print("inside_mask", inside_mask)
	writer.add_scalars('TrainingLoss', { 'Training' : loss_value,}, step)
	for i in range(len(outside_mask)):
		accuracy_total_count += 1
		if inside_mask[i] < 0.55:
			accuracy_inside_count += 1
		if outside_mask[i] < 0.45:
			accuracy_outside_count += 1
		
	if step % 10 == 0:
		item_test = custom_data_loader_test.next()
		input_data_test = item_test["data"]
		mask_test = item_test["mask"]
		output_test = tornado_detection_model(input_data_test)
		loss_test, extra_loss_test_info = loss_function(output_test, mask_test)
		
		writer.add_scalars('Training vs. Testing Loss', { 'Training' : loss_value, 'Testing' : loss_test.item() }, step)
		writer.add_scalars('Accuracy Training', { 'Inside mask' : accuracy_inside_count / accuracy_total_count * 100, 'Outside mask' : accuracy_outside_count / accuracy_total_count * 100 }, step)
		accuracy_total_count = 0
		accuracy_inside_count = 0
		accuracy_outside_count = 0
		
		if step % 50 == 0:
			images = torch.stack([
				mask.cpu()[:16],
				output.cpu()[:16],
				input_data.cpu()[:16,0,0,:,:],
			], 1)
			images.clamp_(0, 1)
			#images = torch.nn.functional.max_pool2d(images, 4)
			img_grid = torchvision.utils.make_grid(images, nrow=4)
			writer.add_image('Training output', img_grid, step)
			#matplotlib_imshow(img_grid, one_channel=False)
			
			# output one image per channel to find bad input processing
			# for i in range(5):
			# 	for l in range(8):
			# 		for j in [1, -1]:
			# 			images = torch.stack([
			# 				mask.cpu()[:16],
			# 				output.cpu()[:16],
			# 				input_data.cpu()[:16,i,l,:,:] * j * 0.5,
			# 			], 1)
			# 			images.clamp_(0, 1)
			# 			#images = torch.nn.functional.max_pool2d(images, 4)
			# 			img_grid = torchvision.utils.make_grid(images, nrow=4)
			# 			writer.add_image('Training output ' + ("-" if j < 0 else "") + " " + str(i) + " layer " + str(l), img_grid, step)
			
			images = torch.stack([
				mask_test.cpu()[:16],
				output_test.cpu()[:16],
				input_data_test.cpu()[:16,0,0,:,:],
			], 1)
			images.clamp_(0, 1)
			#images = torch.nn.functional.max_pool2d(images, 4)
			img_grid = torchvision.utils.make_grid(images, nrow=4)
			writer.add_image('Testing output', img_grid, step)
		if step % 500 == 0 and step != 0:
			print("saving model at step", step)
			saved_data = {
				"state_dict_model": tornado_detection_model.state_dict(),
				"state_dict_optimizer": optimizer.state_dict(),
				"step": step
			}
			torch.save(saved_data, "saved_model.pt.tmp")
			os.rename("saved_model.pt.tmp", "saved_model.pt")
			del saved_data
		if step % 100 == 0:
			torch.cuda.empty_cache()
	writer.flush()
		
	tornado_detection_model.train(True)
	
torch.cuda.empty_cache()
