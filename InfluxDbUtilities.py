## We will use Python client API to write, fetch data
## Install, "pip install influxdb3-python pandas"
import os
from influxdb_client_3 import (
  InfluxDBClient3, InfluxDBError, Point, WritePrecision,
  WriteOptions, write_client_options)

from dotenv import load_dotenv
import os

# Load the .env file
load_dotenv();

host = os.getenv('INFLUX_HOST')
token = os.getenv('INFLUX_TOKEN')
database = os.getenv('INFLUX_DATABASE')

client = InfluxDBClient3(host=host,
                        database=database,
                        token=token,
                        )
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

wco = write_client_options(success_callback=success,
                            error_callback=error,
                            retry_callback=retry,
                            write_options=write_options)


def write_data(table_name,tags,fields,unixTime):
    '''
    Table_name : string.
    tags: dict,
    Fields: dict,
    unixTime: Epoch time.
    '''

    points = {
          "measurement": table_name,
          "tags": tags,
          "fields": fields,
          "time": unixTime
          }    
    
    client.write(points);



def query_data(query):
    '''
    Query : sql query. 
    '''
    table = client.query("SELECT * from home WHERE time >= now() - INTERVAL '90 days'")








