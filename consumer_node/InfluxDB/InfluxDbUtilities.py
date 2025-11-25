## We will use Python client API to write, fetch data
## Install, "pip install influxdb3-python pandas"

import os
from influxdb_client_3 import (
  InfluxDBClient3, InfluxDBError, Point, WritePrecision,
  WriteOptions, write_client_options)

from datetime import datetime
from listeners.sliding_window_listener import SlidingWindowListener

from dotenv import load_dotenv
import os

from datetime import datetime


# Load the .env file
load_dotenv();

def success(self, data: str):
    print(f"Successfully wrote batch: data: {data}")

def error(self, data: str, exception: InfluxDBError):
    print(f"Failed writing batch: config: {self}, data: {data} due: {exception}")

def retry(self, data: str, exception: InfluxDBError):
    print(f"Failed retry writing batch: config: {self}, data: {data} retry: {exception}")


class DatabaseWriter(SlidingWindowListener):

    def __init__(self,verbose = False):
        # Print the path to the .env file being used
        env_path = os.path.abspath('.env')
        
        print(f"Using .env file at: {env_path}")
        self.host = os.getenv('INFLUX_HOST')
        self.token = os.getenv('INFLUX_TOKEN')
        self.database = os.getenv('INFLUX_DATABASE')
        self.verbose = verbose;
        print(f"HOST = {self.host}, TOKEN = {self.token}, DATABASE = {self.database}");

        self.client = InfluxDBClient3(host=self.host,
                                database=self.database,
                                token=self.token,
                                );
        

        
        ## writing options
        self.write_options = WriteOptions(batch_size=500,
                                            flush_interval=10_000,
                                            jitter_interval=2_000,
                                            retry_interval=5_000,
                                            max_retries=5,
                                            max_retry_delay=30_000,
                                            exponential_base=2)
        
                
        wco = write_client_options(success_callback=success,
                                    error_callback=error,
                                    retry_callback=retry,
                                    write_options=self.write_options)
        





    def create_all_measurements(self):
        measurements = ["environment", "anomaly", "trend"]
        for table_name in measurements:
            Measurement_time = "01/01/1970 00.00.01"
            unixTime = int(datetime.strptime(
                Measurement_time, "%d/%m/%Y %H.%M.%S"
            ).timestamp() * 1e9)

            dummy_tags = {"init": "true"}
            dummy_fields = {"dummy_value": 0}

            dummy_point = {
                "measurement": table_name,
                "tags": dummy_tags,
                "fields": dummy_fields,
                "time": unixTime
            }

            self.client.write(dummy_point)

        print("Measurements created (dummy rows inserted).")
    


        

    def write_anomaly(self, anomalous_sample, types, topic):
        '''
        data should have the same structure as other stream data + anomaly type list + topic name (e.g: co_gt,...).

        '''
        self.write_data("anomaly", {"topic": topic},
                        {
                            "value": anomalous_sample['value'],
                            "missing": "missing" in types,
                            "local": "local" in types,
                            "global": "global" in types,
                        }, anomalous_sample['key'])
    
    


    def write_trend(self, sample, type, topic):
        '''
        data should have the same structure as other stream data + trend type + topic name (e.g: co_gt,...).

        '''
        self.write_data("trend", {"topic": topic},
                        {
                            "value": sample['value'],
                            "type": type,
                        }, sample['key'])
    


    def write_quantiles(self, quantiles, topic, key):

        self.write_data("quantiles", {"topic": topic}, {"min": quantiles['min'],
                                                      "q1": quantiles['q1'], "median": quantiles['median'],
                                                      "q3": quantiles['q3'], "max": quantiles['max']}, key
                        )

    def write_moving_statistics(self, stats, topic, key):

        self.write_data("moving_statistics", {"topic": topic}, 
                        {"mean": stats['mean'],
                        "variance": stats['variance']
                        }, key)




    def on_new_window_co_gt(self, data):
        if self.verbose:
            print("++++++++++++++++++++++++++++++++++++++ INFLUXDB +++++++++++++++++++++++++++++++++");
            print(f"len={len(data)} | " + ", ".join(f"{item['key']}: {item['value']}" for item in data))

        last_item = data[-1]
        self.write_data("environment",{"topic":"co_gt"},{"value":last_item['value']},last_item['key'])




    def on_new_window_pt08_s1_co(self, data):
        if self.verbose:
            print("++++++++++++++++++++++++++++++++++++++ INFLUXDB +++++++++++++++++++++++++++++++++");
            print(f"len={len(data)} | " + ", ".join(f"{item['key']}: {item['value']}" for item in data))
        last_item = data[-1]
        self.write_data("environment",{"topic":"pt08_s1_co"},{"value":last_item['value']},last_item['key'])



    def on_new_window_nmhc_gt(self, data):
        if self.verbose:
            print("++++++++++++++++++++++++++++++++++++++ INFLUXDB +++++++++++++++++++++++++++++++++");
            print(f"len={len(data)} | " + ", ".join(f"{item['key']}: {item['value']}" for item in data))
        last_item = data[-1]
        self.write_data("environment",{"topic":"nmhc_gt"},{"value":last_item['value']},last_item['key'])

    def on_new_window_c6h6_gt(self, data):
        if self.verbose:
            print("++++++++++++++++++++++++++++++++++++++ INFLUXDB +++++++++++++++++++++++++++++++++");
            print(f"len={len(data)} | " + ", ".join(f"{item['key']}: {item['value']}" for item in data))
        last_item = data[-1]
        self.write_data("environment",{"topic":"c6h6_gt"},{"value":last_item['value']},last_item['key'])

    def on_new_window_pt08_s2_nmhc(self, data):
        if self.verbose:
            print("++++++++++++++++++++++++++++++++++++++ INFLUXDB +++++++++++++++++++++++++++++++++");
            print(f"len={len(data)} | " + ", ".join(f"{item['key']}: {item['value']}" for item in data))
        last_item = data[-1]
        self.write_data("environment",{"topic":"pt08_s2_nmhc"},{"value":last_item['value']},last_item['key'])

    def on_new_window_nox_gt(self, data):
        if self.verbose:
            print("++++++++++++++++++++++++++++++++++++++ INFLUXDB +++++++++++++++++++++++++++++++++");
            print(f"len={len(data)} | " + ", ".join(f"{item['key']}: {item['value']}" for item in data))
        last_item = data[-1]
        self.write_data("environment",{"topic":"nox_gt"},{"value":last_item['value']},last_item['key'])

    def on_new_window_pt08_s3_nox(self, data):
        if self.verbose:
            print("++++++++++++++++++++++++++++++++++++++ INFLUXDB +++++++++++++++++++++++++++++++++");
            print(f"len={len(data)} | " + ", ".join(f"{item['key']}: {item['value']}" for item in data))
        last_item = data[-1]
        self.write_data("environment",{"topic":"pt08_s3_nox"},{"value":last_item['value']},last_item['key'])

    def on_new_window_no2_gt(self, data):
        if self.verbose:
            print("++++++++++++++++++++++++++++++++++++++ INFLUXDB +++++++++++++++++++++++++++++++++");
            print(f"len={len(data)} | " + ", ".join(f"{item['key']}: {item['value']}" for item in data))
        last_item = data[-1]
        self.write_data("environment",{"topic":"no2_gt"},{"value":last_item['value']},last_item['key'])

    def on_new_window_pt08_s4_no2(self, data):
        if self.verbose:
            print("++++++++++++++++++++++++++++++++++++++ INFLUXDB +++++++++++++++++++++++++++++++++");
            print(f"len={len(data)} | " + ", ".join(f"{item['key']}: {item['value']}" for item in data))
        last_item = data[-1]
        self.write_data("environment",{"topic":"pt08_s4_no2"},{"value":last_item['value']},last_item['key'])

    def on_new_window_pt08_s5_o3(self, data):
        if self.verbose:
            print("++++++++++++++++++++++++++++++++++++++ INFLUXDB +++++++++++++++++++++++++++++++++");
            print(f"len={len(data)} | " + ", ".join(f"{item['key']}: {item['value']}" for item in data))
        last_item = data[-1]
        self.write_data("environment",{"topic":"pt08_s5_o3"},{"value":last_item['value']},last_item['key'])

    def on_new_window_t(self, data):
        if self.verbose:
            print("++++++++++++++++++++++++++++++++++++++ INFLUXDB +++++++++++++++++++++++++++++++++");
            print(f"len={len(data)} | " + ", ".join(f"{item['key']}: {item['value']}" for item in data))
        last_item = data[-1]
        self.write_data("environment",{"topic":"t"},{"value":last_item['value']},last_item['key'])

    def on_new_window_ah(self, data):
        if self.verbose:
            print("++++++++++++++++++++++++++++++++++++++ INFLUXDB +++++++++++++++++++++++++++++++++");
            print(f"len={len(data)} | " + ", ".join(f"{item['key']}: {item['value']}" for item in data))
        last_item = data[-1]
        self.write_data("environment",{"topic":"ah"},{"value":last_item['value']},last_item['key'])

    def on_new_window_rh(self, data):
        if self.verbose:
            print("++++++++++++++++++++++++++++++++++++++ INFLUXDB +++++++++++++++++++++++++++++++++");
            print(f"len={len(data)} | " + ", ".join(f"{item['key']}: {item['value']}" for item in data))
        last_item = data[-1]
        self.write_data("environment",{"topic":"rh"},{"value":last_item['value']},last_item['key'])



            
    # def write_data(self,table_name,tags,fields,Measurement_time):
    #     '''
    #     Table_name : string.
    #     tags: dict,
    #     Fields: dict,
    #     unixTime: Epoch time.
    #     '''
    #     # unixTime = int(datetime.strptime(Measurement_time, "%d/%m/%Y %H.%M.%S").timestamp() * 1e9);
    #     unixTime = int(datetime.strptime(Measurement_time, "%d/%m/%Y %H.%M.%S").timestamp() * 1e6);

    #     print(f"Writing to InfluxDB: measurement={table_name}, tags={tags}, fields={fields}, time={Measurement_time} (unix: {unixTime})");
    #     # for key, val in fields.items():
    #     #     print(f"Field '{key}': value = {val}, type = {type(val)}")

    #     for key, val in fields.items():
    #         if isinstance(val, str):
    #             try:
    #                 fields[key] = float(val)
    #             except ValueError:
    #                 # Keep as string if it's not a number
    #                 pass

                
    #     cleaned_fields = {
    #     k: round(v, 10) if isinstance(v, float) else v 
    #     for k, v in fields.items()
    #     }


    #     try:
    #         points = {
    #             "measurement": table_name,
    #             "tags": tags,
    #             "fields": fields,
    #             "time": unixTime
    #             }    
    #         self.client.write(points);
        
    #     except Exception as e:
    #         print(f"InfluxDB write failed: {e}")
    #         print(f"Point: {points}")
    #         raise
        
    def write_data(self, table_name, tags, fields, Measurement_time):
        '''
        Table_name : string.
        tags: dict,
        Fields: dict,
        Measurement_time: string in format "%d/%m/%Y %H.%M.%S"
        '''
        # from influxdb_client_3 import Point
        
        unixTime = int(datetime.strptime(Measurement_time, "%d/%m/%Y %H.%M.%S").timestamp() * 1e9);

        print(f"Writing to InfluxDB: measurement={table_name}, tags={tags}, fields={fields}, time={Measurement_time} (unix: {unixTime})")

        # Convert string numbers to floats and clean precision
        cleaned_fields = {}
        for key, val in fields.items():
            if isinstance(val, str):
                try:
                    cleaned_fields[key] = round(float(val), 10)
                except ValueError:
                    # Keep as string if it's not a number
                    cleaned_fields[key] = val
            elif isinstance(val, (int, float)):
                # Force everything to float (not int) to avoid 'i' suffix
                cleaned_fields[key] = round(float(val), 10)
            else:
                cleaned_fields[key] = val

        print(f"DEBUG - Cleaned fields: {cleaned_fields}")
        print(f"DEBUG - Field types: {[(k, type(v).__name__) for k, v in cleaned_fields.items()]}")

        try:
            # Use InfluxDB 3 Point API
            point = Point(table_name)
            
            # Add tags
            for key, value in tags.items():
                point = point.tag(key, str(value))
            
            # Add fields
            for key, value in cleaned_fields.items():
                point = point.field(key, value)
            
            # Add timestamp
            point = point.time(unixTime)
            
            # Debug: print the line protocol
            try:
                line_protocol = point.to_line_protocol()
                print(f"DEBUG - Line protocol: {line_protocol}")
            except Exception as lp_error:
                print(f"DEBUG - Could not generate line protocol: {lp_error}")
            
            self.client.write(database=self.database, record=point)
        
        except Exception as e:
            print(f"InfluxDB write failed: {e}")
            print(f"Point data: measurement={table_name}, tags={tags}, fields={cleaned_fields}, time={unixTime}")
            raise



    def query_data(self,query):

        '''
        Query : sql query. 
        returns a table.
        '''

        table = self.client.query(query);
        return table;
    





