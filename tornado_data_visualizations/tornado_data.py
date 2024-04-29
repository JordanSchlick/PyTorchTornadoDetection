import pandas
import pytz
import glob
import dateutil
import re
import os
import time
import math
import threading
import numpy as np

tornado_dataframe = None
tornado_dataframe_lock = threading.Lock()

# print(pandas.to_datetime(dateutil.parser.parse("1970-01-01 00:00:00 -6"), utc=True))
# print(pandas.to_datetime(dateutil.parser.parse("1970-01-01 00:00:00 UTC-6"), utc=True))
# print(pandas.to_datetime("1970-01-01 00:00:00 UTC-6", utc=True))
# print(pandas.Timestamp("1970-01-01 00:00:00 UTC+0").tz_convert("US/Central"))



def load_data():
	"""
	Loads and cleans data about tornados
	"""
	tornado_dataframe_lock.acquire()
	global tornado_dataframe
	try:
		cache = pandas.read_csv("./cache_data_tornados.csv")
		cache.BEGIN_DATE_TIME = pandas.to_datetime(cache.BEGIN_DATE_TIME)
		cache.END_DATE_TIME = pandas.to_datetime(cache.END_DATE_TIME)
		tornado_dataframe = cache
		tornado_dataframe.set_index("BEGIN_TIME_UNIX", drop=False, inplace=True)
		tornado_dataframe.sort_index(inplace=True)
		#tornado_dataframe.info()
		print("Finished loading",tornado_dataframe.shape[0],"entries from cache_data_tornados.csv")
		tornado_dataframe_lock.release()
		return
	except:
		pass
	tornado_dataframe = None
	
	files = glob.glob("../data/StormEvents/StormEvents*")
	#files = ["./data/StormEvents_details-ftp_v1.0_d2013_c20230118.csv.gz"]
	time_regex = re.compile("[+-][0-9.]+")
	suffixes = {"K": 1000, "M": 1000000, "B": 1000000000, "T": 1000000000000}
	suffix_keys = suffixes.keys()
	for file in files:
		print(file, "Loading...", end="\r")
		df = pandas.read_csv(file, compression="infer")
		# only tornados
		df = df.loc[df.EVENT_TYPE == "Tornado"]
		# drop any rows missing locations
		df = df.dropna(subset=["BEGIN_LAT","BEGIN_LON","END_LAT","END_LON", "CZ_TIMEZONE"])
		# remove any entries of landspouts and waterspouts
		df = df.loc[df.EVENT_NARRATIVE.str.contains("landspout")==False]
		df = df.loc[df.EVENT_NARRATIVE.str.contains("waterspout")==False]
		# remove some useless data
		df = df.drop(columns=["BEGIN_YEARMONTH","BEGIN_DAY","BEGIN_TIME","END_YEARMONTH","END_DAY","END_TIME","YEAR"])
		df = df.drop(columns=["MAGNITUDE","MAGNITUDE_TYPE","FLOOD_CAUSE","CATEGORY","DATA_SOURCE"])
		# for some reason the parser is backwards when a prefix is included
		modified_timezone = df.CZ_TIMEZONE.map(lambda x: time_regex.findall(x)[0])
		df.BEGIN_DATE_TIME = pandas.to_datetime(df.BEGIN_DATE_TIME + " " + modified_timezone)
		df.END_DATE_TIME = pandas.to_datetime(df.END_DATE_TIME + " " + modified_timezone)
		# convert to meters
		df.TOR_LENGTH = df.TOR_LENGTH * 1609.344
		df.TOR_WIDTH = df.TOR_WIDTH * 0.3048
		# convert to dollars based on suffix
		df.DAMAGE_PROPERTY = df.DAMAGE_PROPERTY.map(lambda x: float(x[:-1]) * suffixes[x[-1]] if type(x) is str and x[-1] in suffix_keys else x)
		df.DAMAGE_CROPS = df.DAMAGE_CROPS.map(lambda x: float(x[:-1]) * suffixes[x[-1]] if type(x) is str and x[-1] in suffix_keys else x)
		#df.DAMAGE_PROPERTY = df.DAMAGE_PROPERTY.map(lambda x: str(type(x)))
		
		
		#print(df.head())
		#print(df.info())
		
		# convert to unix timestamps
		df["BEGIN_TIME_UNIX"] = (pandas.to_datetime(df.BEGIN_DATE_TIME, utc=True) - pandas.Timestamp("1970-01-01 00:00:00 UTC+0")) // pandas.Timedelta('1s')
		df["END_TIME_UNIX"] = (pandas.to_datetime(df.END_DATE_TIME, utc=True) - pandas.Timestamp("1970-01-01 00:00:00 UTC+0")) // pandas.Timedelta('1s')
		df["DURATION"] = df["END_TIME_UNIX"] - df["BEGIN_TIME_UNIX"]
		
		# radar data before May 2013 is likely to not have dual polarization data so remove date before then
		# technically some radar data may not have dual polarization until July of 2013 but those are the minority
		#df = df.loc[df["BEGIN_TIME_UNIX"] > 1367384400]
		
		# there was one error in the dataset that says a tornado was on the ground for over 24 hours
		df = df.loc[df["DURATION"] < 60 * 60 * 6]
		
		#tornado_dataframe["date_interval"] = pandas.Interval(date_unix, date_unix + duration)
		if tornado_dataframe is None:
			tornado_dataframe = df
		else:
			tornado_dataframe = pandas.concat([tornado_dataframe, df])
		print(file, "Loaded    ")
		#print(tornado_dataframe.tail(20))
	#tornado_dataframe.sort_values("BEGIN_TIME_UNIX", inplace=True, inplace=True)
	tornado_dataframe.set_index("BEGIN_TIME_UNIX", drop=False, inplace=True)
	tornado_dataframe.sort_index(inplace=True)
	tornado_dataframe.info()
	print("Finished loading",tornado_dataframe.shape[0],"entries")
	try:
		tornado_dataframe.to_csv("./cache_data_tornados.csv")
	except:
		pass
	tornado_dataframe_lock.release()

def overlaps(a1, a2, b1, b2):
	"""
	Return the amount of overlap, in bp
	between a and b.
	If >0, the number of bp of overlap
	If 0,  they are book-ended.
	If <0, the distance in bp between them
	"""
	return min(a2, b2) - max(a1, b1)
	
# print(overlaps(0,1,1,2))
# print(overlaps(0,1,1.2,2))
# print(overlaps(0,1,0.9,2))
# print(overlaps(0,1,0.5,0.6))
# print(overlaps(0.5,0.6,0,1))

def get_tornados(start_time, end_time):
	"""Get tornados that occur in the given time frame

	Args:
		start_time (number): unix epoch of the start of the time frame
		end_time (number):  unix epoch of the end of the time frame

	Returns:
		pandas.dataframe: A dataframe containing all tornados from the time frame
	"""
	global tornado_dataframe
	if tornado_dataframe is None:
		load_data()
	#tornado_dataframe_lock.acquire()
	# Get tornados with dates close to the times
	df = tornado_dataframe.loc[start_time-24*60*60:end_time+24*60*60]
	# filter remaining elements to be in range
	df = df[df.apply(lambda x: overlaps(x.BEGIN_TIME_UNIX, x.END_TIME_UNIX, start_time, end_time) >= 0, axis=1)]
	#df = list(df.iterrows())
	#tornado_dataframe_lock.release()
	return df

def get_all_tornados():
	"""Gets all tornados in the data

	Returns:
		pandas.dataframe: A dataframe containing all tornados from the data
	"""
	global tornado_dataframe
	if tornado_dataframe is None:
		load_data()
	return tornado_dataframe

def benchmark():
	get_tornados(1370042092.1780014, 1370042343.552)
	start_time = time.time()
	for i in range(1000):
		get_tornados(1370042092.1780014, 1370042343.552)
	duration = time.time() - start_time
	print("duration", duration)

def interpolate_3d(x1, x2, amount):
	return (x1[0] * (1 - amount) + x2[0] * amount, x1[1] * (1 - amount) + x2[1] * amount, x1[2] * (1 - amount) + x2[2] * amount)

def generate_mask(radar_data):
	"""Generates a mask of where tornados are in a radar volume

	Args:
		radar_data (RadarData): radar data object to generate the mask for

	Returns:
		numpy.array: a 2d array with the size of a sweep excluding the padding rays
		list[dict]: a list of dictionaries containing info about tornados
	"""
	stats = radar_data.get_stats()
	# expand time range slightly to get tornados that ate about to happen
	tornados = get_tornados(stats["begin_time"] - 60, stats["end_time"] + 180)
	if len(tornados) > 50:
		print(stats)
		print("to many tornados", len(tornados))
		return []
		#os._exit(1)
		#print(tornados)
	scan_time = stats["begin_time"] + (stats["end_time"] - stats["begin_time"]) / 4
	if len(tornados) == 0:
		return np.zeros((radar_data.theta_buffer_count, radar_data.radius_buffer_count)), []
	mask = None
	theta_buffer_count = radar_data.theta_buffer_count
	radius_buffer_count = radar_data.radius_buffer_count
	X, Y = np.ogrid[:theta_buffer_count, :radius_buffer_count]
	info = []
	for index, tornado in tornados:
		# time in tornados lifespan 0.0 to 1.0
		# can be outside that range if before or after tornado
		tornado_time = (scan_time - tornado.BEGIN_TIME_UNIX) / max((tornado.END_TIME_UNIX - tornado.BEGIN_TIME_UNIX), 30)
		#print(tornado_time, tornado.TOR_F_SCALE)
		
		# get current position in radar space
		begin_pos = radar_data.get_radar_space_for_location(tornado.BEGIN_LAT, tornado.BEGIN_LON, 0)
		end_pos = radar_data.get_radar_space_for_location(tornado.END_LAT, tornado.END_LON, 0)
		pos = interpolate_3d(begin_pos, end_pos, tornado_time)
		
		# get location in buffer
		pixel_info = radar_data.get_pixel_for_radar_space(pos[0], pos[1], pos[2])
		pixel_location_theta = pixel_info["theta"]
		pixel_location_radius = pixel_info["radius"]
		# set radius of 15km based on pixel dimensions
		radius = 15000 / pixel_info["pixel_radius_length"]
		oval_ratio = pixel_info["pixel_theta_width"] / pixel_info["pixel_radius_length"]
		
		if pixel_location_radius >= radius_buffer_count:
			# exclude tornados outside the data
			continue
		
		# TODO: this does not handle wrapping correctly and cuts off the mask when it should wrap around theta
		oval_ratio = oval_ratio**2
		distance = oval_ratio * (X - pixel_location_theta)**2 + (Y - pixel_location_radius)**2
		radius = (radius)**2
		circular_mask = (distance < radius)
		if mask is None:
			mask = circular_mask
		else:
			mask = mask | circular_mask
		
		info.append({
			"rating": tornado.TOR_F_SCALE,
			"tornado_time": tornado_time,
			"location_theta": pixel_location_theta,
			"location_radius": pixel_location_radius,
			"radar_distance": math.sqrt(pos[0]**2 + pos[1]**2 + pos[2]**2),
			"position": pos,
			"original": tornado,
		})
	return mask, info

if __name__ == "__main__":
	load_data()
	# from el reno radar data file
	#print(get_tornados(1370042092.1780014, 1370042343.552))
	#benchmark()

