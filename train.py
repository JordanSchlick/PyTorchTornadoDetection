import time
import math
import multiprocessing
from matplotlib import pyplot as plt
import numpy as np
import torch
import torchvision.utils
import torch.utils.data
import model
import dataset


device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
device = torch.device(device_str)

dataset_files = dataset.DirectoryTrainTest("./data/Radar/l2data", train_percentage=90)
print("Data split", len(dataset_files.train_list), len(dataset_files.test_list))
thread_count = max(math.ceil(multiprocessing.cpu_count() / 2) - 2, 1)
tornado_dataset = dataset.TornadoDataset(dataset_files.test_list, thread_count=thread_count, buffer_size=thread_count * 2, section_size=256, auto_shuffle=True)
torch_tornado_dataset = dataset.TorchDataset(tornado_dataset)
data_loader = torch.utils.data.DataLoader(torch_tornado_dataset, 16) #, num_workers=2, pin_memory=True, pin_memory_device=device_str
data_iter = iter(data_loader)
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

tornado_detection_model = model.TornadoDetectionModel()

pytorch_total_params = sum(p.numel() for p in tornado_detection_model.parameters())
print("parameter count", pytorch_total_params)

tornado_detection_model.to(device)
#optimizer = torch.optim.SGD(tornado_detection_model.parameters(), lr=0.01, momentum=0.9)
optimizer = torch.optim.AdamW(tornado_detection_model.parameters(),lr=0.0001)

loss_function = model.MaskLoss()
loss_function.to(device)




for step in range(1000):
	item = next(data_iter)
	input_data = item["data"]
	input_data = input_data.to(device)
	mask = item["mask"]
	mask = mask.to(device)
	
	print(input_data.shape)
	optimizer.zero_grad()
	
	output = tornado_detection_model(input_data)
	#loss = torch.mean(torch.maximum(1 - output, torch.tensor(0)))
	loss, extra_loss_info = loss_function(output, mask)
	
	loss.backward()
	optimizer.step()
	
	tornado_detection_model.train(False)
	mean_out = torch.mean(output)
	min_out = torch.min(output)
	max_out = torch.max(output)
	print("step", step, "loss", loss.item(), "mean", mean_out.detach().cpu().numpy(), "min", min_out.detach().cpu().numpy(), "max", max_out.detach().cpu().numpy())
	print("outside_mask", extra_loss_info["outside_mask"].detach().cpu().numpy())
	print("inside_mask", extra_loss_info["inside_mask"].detach().cpu().numpy())
	images = torch.stack([
		mask,
		output,
		input_data[:,0,0,:,:],
	], 1)
	images = torch.nn.functional.max_pool2d(images, 4)
	img_grid = torchvision.utils.make_grid(images, nrow=4)
	#matplotlib_imshow(img_grid, one_channel=False)
	tornado_detection_model.train(True)
	
torch.cuda.empty_cache()
