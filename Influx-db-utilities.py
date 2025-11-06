## We will use Python client API to write, fetch data
## Install, "pip install influxdb3-python pandas"
import os
from influxdb_client_3 import (
  InfluxDBClient3, InfluxDBError, Point, WritePrecision,
  WriteOptions, write_client_options)

host = os.getenv('INFLUX_HOST')
token = os.getenv('INFLUX_TOKEN')
database = os.getenv('INFLUX_DATABASE')

## writing options
write_options = WriteOptions(batch_size=500,
                                    flush_interval=10_000,
                                    jitter_interval=2_000,
                                    retry_interval=5_000,
                                    max_retries=5,
                                    max_retry_delay=30_000,
                                    exponential_base=2)


## Callbacks
def success(self, data: str):
    print(f"Successfully wrote batch: data: {data}")

def error(self, data: str, exception: InfluxDBError):
    print(f"Failed writing batch: config: {self}, data: {data} due: {exception}")

def retry(self, data: str, exception: InfluxDBError):
    print(f"Failed retry writing batch: config: {self}, data: {data} retry: {exception}")



def line_protocol_contruction(table_name,tags,fields,unixTime):
    '''
    Fields : A dictionary. {'temp':20.0,...}
    
    '''
    # Create an array of points with tags and fields.
    points = [Point(table_name)
                .tag("room", "Kitchen")
                .field("temp", 25.3)
                .field('hum', 20.2)
                .field('co', 9)]


def write_data(host,token,database,points):
    with InfluxDBClient3(host=host,
                            token=token,
                            database=database,
                            write_client_options=wco) as client:

        client.write(points, write_precision='s')









