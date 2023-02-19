import os
import dateutil
import pandas as pd
import tornado_data as td
import boto3
from botocore.handlers import disable_signing
import math
import datetime as dt

tornado_df = None
stations_df = None
radar_bucket = None
surface_radius = 6371000.0
S3_BUCKET_NAME = "noaa-nexrad-level2"

def load_data():
	global tornado_df
	global stations_df
	global radar_bucket

	stations_df = pd.read_csv("./data/Radar/NEXRAD_Stations.csv")
	stations_df = stations_df.loc[stations_df.STATION_ID.str.contains("NEXRAD:T")==False]
	tornado_df = td.get_all_tornados()
	#tornado_df = tornado_df.loc[tornado_df.BEGIN_TIME_UNIX > 1577858400]
	print("All hail the rat god!")
	
	# create the aws s3 link to 'noaa-nexrad-level2' bucket
	s3object = boto3.resource("s3")
	s3object.meta.client.meta.events.register('choose-signer.s3.*', disable_signing)
	radar_bucket = s3object.Bucket(S3_BUCKET_NAME)

# finds the distance between two longitudinal and latitudinal coordinates
def find_distance(lat1, lon1, lat2, lon2):
	lat1 = math.radians(lat1)
	lon1 = math.radians(lon1)
	lat2 = math.radians(lat2)
	lon2 = math.radians(lon2)
	dlon = lon2 - lon1
	dlat = lat2 - lat1
	a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
	c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
	return surface_radius * c

# checks the given file name to see if it is within the given time frame
def download_file_check(file_name, date, station_name, begin_unix_time, end_unix_time):
	date_sections = file_name.replace((date + '/' + station_name + '/' + station_name), '').split('_')
	d = date_sections[0][0:4] + "-" + date_sections[0][4:6] + "-" + date_sections[0][6:]
	t = date_sections[1][0:2] + ":" + date_sections[1][2:4] + ":" + date_sections[1][4:]
	unix_time = dateutil.parser.parse((d + ' ' + t + ' +0')).timestamp()
	if unix_time > begin_unix_time and unix_time < end_unix_time:
		# print("File Time: " + str(dt.datetime.utcfromtimestamp(unix_time)))
		return True
	else:
		return False

# acquires and caches the file list from the nexrad aws s3 bucket
def find_l2_files(prefix):
	path = "./data/Radar/s3cache/" + prefix.replace('/', '-')
	file_names = []
	if os.path.isfile(path):
		file = open(path, "r")
		file_names = file.read().split('\n')
		file.close()
		if file_names[-1] == '':
			file_names = file_names[:-1]
	else:
		print("Getting list for prefix " + prefix)
		bucket_data = radar_bucket.objects.filter(Prefix=prefix)
		file = open(path, "w")
		file_names = list(map(lambda x: x.key, bucket_data))
		file.write("\n".join(file_names))
		file.close()
	return file_names

# downloads and moves all the files in the list
def download_l2_files(files_to_download):
	global radar_bucket
	for obj in files_to_download:
		if obj.split('_')[-1] != 'MDM':
			file_name = obj.split('/')[-1]
			file_path = "./data/Radar/temp/" + file_name
			new_file_path = "./data/Radar/l2data/" + file_name
			if not os.path.isfile(new_file_path):
				# with open(file_path, 'w') as fp:
				# 	pass
				radar_bucket.download_file(obj, file_path)
				print("Download of " + file_name + " completed")
				os.rename(file_path, new_file_path)
				print("Moving of " + file_name + " from './data/Radar/temp/' to './data/Radar/l2data/' completed")
			else:
				print("File " + file_name + " already is downloaded")
		else:
			print("We don't know what MDM files do, so we will ignore them")

def download_nearest_operational_radar_data(tornado, stations_to_remove = None):
	global stations_df
	global radar_bucket

	avg_lat = (tornado["BEGIN_LAT"] + tornado["END_LAT"])/2
	avg_lon = (tornado["BEGIN_LON"] + tornado["END_LON"])/2

	temp_stations_df = stations_df
	
	if stations_to_remove is None:
		stations_to_remove = []
	
	# remove stations that were down during the time (based on previous runs of the function)
	if stations_to_remove != []:
		for station in stations_to_remove:
			temp_stations_df.drop(temp_stations_df.index[temp_stations_df['STATION_ID'] == station], axis = 0, inplace = True)
			print("Station " + station + " removed from dataframe")

	distances = []
	for station in stations_df.iterrows():
		distances.append(find_distance(avg_lat, avg_lon, station[1]['LATITUDE'], station[1]['LONGITUDE']))

	temp_stations_df['DISTANCE'] = distances
	temp_stations_df = temp_stations_df.loc[temp_stations_df['DISTANCE'] < 400000]
	
	if temp_stations_df.empty:
		# no radars available
		return
		
	
	ans = temp_stations_df.iloc[(temp_stations_df['DISTANCE']).abs().argsort()[:1]]
	print("Station Found: ")
	print(ans)

	begin_date_time = dt.datetime.utcfromtimestamp(tornado['BEGIN_TIME_UNIX'])
	begin_date_time -= dt.timedelta(minutes=6) # move start time back to ensure the volume where the tornado is forming is downloaded
	begin_date = str(begin_date_time.date()).replace('-', '/')

	end_date_time = dt.datetime.utcfromtimestamp(tornado['END_TIME_UNIX'])
	end_date = str(end_date_time.date()).replace('-', '/')

	station_id = str(ans['STATION_ID'])
	station_name_index = station_id.find(":") + 1
	station_name = station_id[station_name_index:(station_name_index + 4)]

	prefix = begin_date + '/' + station_name + '/' + station_name + begin_date.replace('/', '')
	alt_prefix = end_date + '/' + station_name + '/' + station_name + end_date.replace('/', '')

	files_to_download = []
	# print("Begin Time: " + str(begin_date_time))
	for file_name in find_l2_files(prefix):
		if download_file_check(file_name, begin_date, station_name, dateutil.parser.parse(begin_date_time.strftime("%Y-%m-%d %H:%M:%S") + ' +0').timestamp(), dateutil.parser.parse(end_date_time.strftime("%Y-%m-%d %H:%M:%S") + ' +0').timestamp()):
			files_to_download.append(file_name)
	if begin_date != end_date:
		for file_name in find_l2_files(alt_prefix):
			if download_file_check(file_name, end_date, station_name, dateutil.parser.parse(begin_date_time.strftime("%Y-%m-%d %H:%M:%S") + ' +0').timestamp(), dateutil.parser.parse(end_date_time.strftime("%Y-%m-%d %H:%M:%S") + ' +0').timestamp()):
				files_to_download.append(file_name)
	# print("End Time: " + str(end_date_time))

	print("Files to Download: ")
	print(files_to_download)

	if files_to_download == []:
		stations_to_remove.append("NEXRAD:" + station_name)
		download_nearest_operational_radar_data(tornado, stations_to_remove)

	download_l2_files(files_to_download)

def get_radar_data():
	global tornado_df
	load_data()
	try:
		os.mkdir('./data/Radar/l2data')
	except OSError as err:
		pass
	try:
		os.mkdir('./data/Radar/s3cache')
	except OSError as err:
		pass
	try:
		os.mkdir('./data/Radar/temp')
	except OSError as err:
		pass
	for tornado in tornado_df.iterrows():
		download_nearest_operational_radar_data(tornado[1])
		#break
		

get_radar_data()