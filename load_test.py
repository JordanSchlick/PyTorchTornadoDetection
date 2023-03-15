import time
import math
import multiprocessing
from matplotlib import pyplot as plt
import numpy as np
import importlib
import dataset
importlib.reload(dataset)

dataset_files = dataset.DirectoryTrainTest("./data/Radar/l2data", train_percentage=90)
print("Split", len(dataset_files.train_list), len(dataset_files.test_list))
thread_count = max(math.ceil(multiprocessing.cpu_count() / 2) - 2, 1)
files = dataset_files.test_list + dataset_files.train_list
tornado_dataset = dataset.TornadoDataset(files , thread_count=thread_count, buffer_size=thread_count*2, section_size=256, cache_results=True, ignore_cache=True)
#time.sleep(10)
for i in range(100000):
	data = tornado_dataset.next()
	data["data"]=data["data"].shape
	data["mask"]=data["mask"].shape
	data["tornado_info"]["narrative"]="expunged"
	print(data)
tornado_dataset.destroy()