import pandas
import pytz

tornado_dataframe = None


#print(pandas.to_datetime("1970-01-01 00:00:00 UTC-6", utc=True))
#print(pandas.Timestamp("1970-01-01 00:00:00 UTC+0").tz_convert("US/Central"))

def load_data():
	global tornado_dataframe
	df = pandas.read_csv("./1950-2021_torn.csv")
	df = df.loc[(df.tz == 3)] # almost all of them are time zone 3
	#print(df.head())
	tornado_dataframe = pandas.DataFrame({
		"date": pandas.to_datetime(df.date + " " + df.time + " UTC-6", utc=True),
		"tornado_number": df.om,
		"state": df.st,
		"magnitude": df.mag,
		"injuries": df.inj,
		"fatalities": df.fat,
		"property_loss": df.loss,
		"crop_loss": df.closs,
		"start_latitude": df.slat,
		"start_longitude": df.slon,
		"end_latitude": df.elat,
		"end_longitude": df.elon,
		"length": df.len * 1609.344, # meters
		"width": df.wid / 0.9144, # meters
	})
	# estimate duration of tornado
	speed = 30 / 2.237 # m/s
	duration = tornado_dataframe.length / speed
	duration = duration.clip(lower=120)
	tornado_dataframe["duration"] = duration
	
	date_unix = (tornado_dataframe.date - pandas.Timestamp("1970-01-01 00:00:00 UTC+0")) // pandas.Timedelta('1s')
	tornado_dataframe["date_unix"] = date_unix
	
	#tornado_dataframe["date_interval"] = pandas.Interval(date_unix, date_unix + duration)
	print(tornado_dataframe.tail(20))

load_data()

def get_tornados(startTime, endTime):
	global tornado_dataframe
	if tornado_dataframe is None:
		load_data()
	tornados = []

def generate_mask(radarData):
	stats = radarData.get_stats()
		
	