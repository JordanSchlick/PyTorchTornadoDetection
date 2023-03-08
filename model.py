import torch
import torch.functional

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