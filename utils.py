# import tensorflow as tf
import os
import math
import random
import numpy

# # possible alow memory to be allocated and freed more effectively
# os.environ["TF_GPU_ALLOCATOR"]="cuda_malloc_async"

# # initializes tensorflow to use memory growth and a set maximum
# def tf_init(max_memory_gb = 3):
# 	gpus = tf.config.experimental.list_physical_devices('GPU')
# 	if gpus:
# 		try:
# 			# Currently, memory growth needs to be the same across GPUs
# 			for gpu in gpus:
# 				tf.config.experimental.set_memory_growth(gpu, True)
# 				if max_memory_gb is not None:
# 					tf.config.experimental.set_virtual_device_configuration(gpu,
# 					[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024*max_memory_gb)])
# 			logical_gpus = tf.config.experimental.list_logical_devices('GPU')
# 			print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
# 		except RuntimeError as e:
# 			# Memory growth must be set before GPUs have been initialized
# 			print(e)



# def first_in_batch(data_t):
# 	data_t=tf.unstack(data_t)[0]
# 	return data_t