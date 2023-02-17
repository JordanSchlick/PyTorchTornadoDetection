import dateutil
import pandas as pd
import tornado_data as td
import boto3
from botocore.handlers import disable_signing
import math
import datetime as dt
import time

tornado_df = None
stations_df = None
radar_bucket = None
surface_radius = 6371000.0

def load_data():
	global tornado_df
	global stations_df
	global radar_bucket
	stations_df = pd.read_csv("./data/Radar/NEXRAD_Stations.csv")
	stations_df = stations_df.loc[stations_df.STATION_ID.str.contains("NEXRAD:T")==False]
	tornado_df = td.get_all_tornados()
	print("All hail the rat god!")
	
	S3_BUCKET_NAME = "noaa-nexrad-level2"
	s3_resource = boto3.resource("s3")
	s3_resource.meta.client.meta.events.register('choose-signer.s3.*', disable_signing)
	radar_bucket = s3_resource.Bucket(S3_BUCKET_NAME)
	# print('Listing Amazon S3 Bucket objects/files:')
	# for obj in radar_bucket.objects.filter(Prefix='2018'):
	#     print(f'-- {obj.key}')

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
	
def download_file_check(obj, date, station_name, begin_unix_time, end_unix_time):
	date_sections = str(obj.key).replace((date + '/' + station_name + '/' + station_name), '').split('_')
	d = date_sections[0][0:4] + "-" + date_sections[0][4:6] + "-" + date_sections[0][6:]
	t = date_sections[1][0:2] + ":" + date_sections[1][2:4] + ":" + date_sections[1][4:]
	unix_time = dateutil.parser.parse((d + ' ' + t + ' +0')).timestamp()
	if unix_time > begin_unix_time and unix_time < end_unix_time:
		print("File Time: " + str(dt.datetime.utcfromtimestamp(unix_time)))
		return True
	else:
		return False

def download_nearest_operational_radar_data(tornado, stations_to_remove = []):
	global stations_df
	global radar_bucket

	avg_lat = (tornado["BEGIN_LAT"] + tornado["END_LAT"])/2
	avg_lon = (tornado["BEGIN_LON"] + tornado["END_LON"])/2

	temp_stations_df = stations_df

	# remove stations that were down during the time (based on previous runs of the function)
	if stations_to_remove != []:
		for station in stations_to_remove:
			temp_stations_df.drop(temp_stations_df.index[temp_stations_df['STATION_ID'] == station], axis = 0, inplace = True)
			print("Station " + station + " removed from dataframe")

	distances = []
	for station in stations_df.iterrows():
		distances.append(find_distance(avg_lat, avg_lon, station[1]['LATITUDE'], station[1]['LONGITUDE']))

	temp_stations_df['DISTANCE'] = distances
	ans = temp_stations_df.iloc[(temp_stations_df['DISTANCE']).abs().argsort()[:1]]
	print("Station Found: ")
	print(ans)

	begin_date_time = dt.datetime.utcfromtimestamp(tornado['BEGIN_TIME_UNIX'])
	begin_date_time -= dt.timedelta(minutes=6)
	begin_date = str(begin_date_time.date()).replace('-', '/')

	end_date_time = dt.datetime.utcfromtimestamp(tornado['END_TIME_UNIX'])
	end_date = str(end_date_time.date()).replace('-', '/')

	station_id = str(ans['STATION_ID'])
	station_name_index = station_id.find(":") + 1
	station_name = station_id[station_name_index:(station_name_index + 4)]

	prefix = begin_date + '/' + station_name + '/' + station_name + begin_date.replace('/', '') + '_'
	alt_prefix = end_date + '/' + station_name + '/' + station_name + end_date.replace('/', '') + '_'

	file_to_download = []
	print("Begin Time: " + str(begin_date_time))
	for obj in radar_bucket.objects.filter(Prefix=prefix):
		if download_file_check(obj, begin_date, station_name, dateutil.parser.parse(begin_date_time.strftime("%Y-%m-%d %H:%M:%S") + ' +0').timestamp(), tornado['END_TIME_UNIX']):
			file_to_download.append(obj.key)
	if begin_date != end_date:
		for obj in radar_bucket.objects.filter(Prefix=alt_prefix):
			if download_file_check(obj, end_date, station_name, dateutil.parser.parse(begin_date_time.strftime("%Y-%m-%d %H:%M:%S") + ' +0').timestamp(), tornado['END_TIME_UNIX']):
				file_to_download.append(obj.key)
	print("End Time: " + str(end_date_time))
	print("Files to Download: ")
	print(file_to_download)

	if file_to_download == []:
		stations_to_remove.append("NEXRAD:" + station_name)
		download_nearest_operational_radar_data(tornado, stations_to_remove)

	

	
def download_radar_data():
	global tornado_df
	load_data()
	for tornado in tornado_df.iterrows():
		download_nearest_operational_radar_data(tornado[1])
		break
		


download_radar_data()