import torch
import torch.functional
import torchvision

class TornadoDetectionModel(torch.nn.Module):
	def __init__(self):
		super().__init__()
		sweep_count = 8
		self.conv1 = torch.nn.Conv3d(5, 32, 3, padding="same")
		self.conv2 = torch.nn.Conv2d(32 * sweep_count, 64, 3, padding="same")
		self.conv3 = torch.nn.Conv2d(64, 32, 7, padding="same")
		self.conv4 = torch.nn.Conv2d(32, 16, 5, padding="same")
		self.conv5 = torch.nn.Conv2d(16, 1, 3, padding="same")
	
	def forward(self, x):
		x = torch.nn.functional.leaky_relu(self.conv1(x))
		# combine channels and sweeps into one dimension
		x = x.view(x.shape[0], -1, x.shape[3], x.shape[4])
		x = torch.nn.functional.leaky_relu(self.conv2(x))
		x = torch.nn.functional.leaky_relu(self.conv3(x))
		x = torch.nn.functional.leaky_relu(self.conv4(x))
		x = torch.sigmoid(self.conv5(x))
		x = torch.squeeze(x, axis=1)
		return x

class MaskLoss(torch.nn.Module):
	"""
	A loss function for finding tornados.
	The space inside the mask should have a high value somewhere, while the outside should be zero.
	The output loss is between 0 and 2.5
	Loss values less than 0.5 indicate the model is able to differentiate between the inside and outside.
	"""
	def __init__(self):
		super().__init__()

	def forward(self, output, mask):
		# crop edges that have problems due to padding
		output = torchvision.transforms.functional.crop(output, 8, 8, output.shape[1] - 16, output.shape[2] - 16)
		output = output.reshape(output.shape[0], -1)
		mask = torchvision.transforms.functional.crop(mask, 8, 8, mask.shape[1] - 16, mask.shape[2] - 16)
		mask = mask.reshape(mask.shape[0], -1)
		
		# calculate loss outside of mask
		outside_mask = torch.abs(output * (1 - mask))
		#print(output.shape, output.dtype, mask.shape, mask.dtype, outside_mask.shape, outside_mask.dtype)
		outside_mask = torch.mean(outside_mask, dim=-1) * 0.5 + torch.max(outside_mask, dim=-1).values * 0.5
		outside_mask_mean = torch.mean(outside_mask)
		
		# calculate loss for inside of mask
		inside_mask = output * mask
		# mean is included inside the mask to keep gradients alive
		inside_mask = 1 - (torch.max(inside_mask, dim=-1).values * 0.9 + torch.mean(inside_mask, dim=-1) * 0.1)
		inside_mask_mean = torch.mean(inside_mask)
		
		# punish optimizing for only one part of mask
		imbalance = torch.abs(inside_mask_mean - outside_mask_mean)
		imbalance = imbalance * imbalance * imbalance
		
		loss = outside_mask_mean + inside_mask_mean * 1.5 + imbalance * 2
		#print(loss)
		return torch.mean(loss), {
			"outside_mask": outside_mask,
			"inside_mask": inside_mask
		}