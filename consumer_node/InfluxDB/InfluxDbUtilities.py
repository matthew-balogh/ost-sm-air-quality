## We will use Python client API to write, fetch data
## Install, "pip install influxdb3-python pandas"

import os
from influxdb_client_3 import (
  InfluxDBClient3, InfluxDBError, Point, WritePrecision,
  WriteOptions, write_client_options)

from datetime import datetime, timedelta
from listeners.sliding_window_listener import SlidingWindowListener

from dotenv import load_dotenv
import os

# Load the .env file
load_dotenv();

def success(self, data: str):
    print(f"Successfully wrote batch: data: {data}")

def error(self, data: str, exception: InfluxDBError):
    print(f"Failed writing batch: config: {self}, data: {data} due: {exception}")

def retry(self, data: str, exception: InfluxDBError):
    print(f"Failed retry writing batch: config: {self}, data: {data} retry: {exception}")


class DatabaseWriter(SlidingWindowListener):

    def __init__(self,verbose = True):
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
        # NOTE: Do not instantiate any forecaster here. Online forecasting
        # is handled by a separate observer (`OnlineForecaster`) which will
        # call `write_prediction()` on this DatabaseWriter when predictions
        # are available. Keeping this class free of forecaster instantiation
        # avoids circular imports and keeps responsibilities separate.
        self.forecaster = None



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



    def on_new_window_pt08_s1_co(self, data):
        if self.verbose:
            print("++++++++++++++++++++++++++++++++++++++ INFLUXDB +++++++++++++++++++++++++++++++++");
            print(f"len={len(data)} | " + ", ".join(f"{item['key']}: {item['value']}" for item in data))
        last_item = data[-1]
        self._write_value_and_predictions(topic='pt08_s1_co', last_item=last_item)


    def on_new_window_pt08_s2_nmhc(self, data):
        if self.verbose:
            print("++++++++++++++++++++++++++++++++++++++ INFLUXDB +++++++++++++++++++++++++++++++++");
            print(f"len={len(data)} | " + ", ".join(f"{item['key']}: {item['value']}" for item in data))
        last_item = data[-1]
        self._write_value_and_predictions(topic='pt08_s2_nmhc', last_item=last_item)

    def on_new_window_pt08_s3_nox(self, data):
        if self.verbose:
            print("++++++++++++++++++++++++++++++++++++++ INFLUXDB +++++++++++++++++++++++++++++++++");
            print(f"len={len(data)} | " + ", ".join(f"{item['key']}: {item['value']}" for item in data))
        last_item = data[-1]
        self._write_value_and_predictions(topic='pt08_s3_nox', last_item=last_item)

    def on_new_window_pt08_s4_no2(self, data):
        if self.verbose:
            print("++++++++++++++++++++++++++++++++++++++ INFLUXDB +++++++++++++++++++++++++++++++++");
            print(f"len={len(data)} | " + ", ".join(f"{item['key']}: {item['value']}" for item in data))
        last_item = data[-1]
        self._write_value_and_predictions(topic='pt08_s4_no2', last_item=last_item)

    def on_new_window_pt08_s5_o3(self, data):
        if self.verbose:
            print("++++++++++++++++++++++++++++++++++++++ INFLUXDB +++++++++++++++++++++++++++++++++");
            print(f"len={len(data)} | " + ", ".join(f"{item['key']}: {item['value']}" for item in data))
        last_item = data[-1]
        self._write_value_and_predictions(topic='pt08_s5_o3', last_item=last_item)

    def on_new_window_t(self, data):
        if self.verbose:
            print("++++++++++++++++++++++++++++++++++++++ INFLUXDB +++++++++++++++++++++++++++++++++");
            print(f"len={len(data)} | " + ", ".join(f"{item['key']}: {item['value']}" for item in data))
        last_item = data[-1]
        self._write_value_and_predictions(topic='t', last_item=last_item)

    def on_new_window_ah(self, data):
        if self.verbose:
            print("++++++++++++++++++++++++++++++++++++++ INFLUXDB +++++++++++++++++++++++++++++++++");
            print(f"len={len(data)} | " + ", ".join(f"{item['key']}: {item['value']}" for item in data))
        last_item = data[-1]
        self._write_value_and_predictions(topic='ah', last_item=last_item)

    def on_new_window_rh(self, data):
        if self.verbose:
            print("++++++++++++++++++++++++++++++++++++++ INFLUXDB +++++++++++++++++++++++++++++++++");
            print(f"len={len(data)} | " + ", ".join(f"{item['key']}: {item['value']}" for item in data))
        last_item = data[-1]
        self._write_value_and_predictions(topic='rh', last_item=last_item)

    def on_new_window_c6h6_gt(self, data):
        pass

    def on_new_window_co_gt(self, data):
        pass

    def on_new_window_nmhc_gt(self, data):
        pass

    def on_new_window_no2_gt(self, data):
        pass

    def on_new_window_nox_gt(self, data):
        pass
    
            
    def write_data(self,table_name,tags,fields,Measurement_time):
        '''
        Table_name : string.
        tags: dict,
        Fields: dict,
        unixTime: Epoch time.
        '''
        unixTime = int(datetime.strptime(Measurement_time, "%d/%m/%Y %H.%M.%S").timestamp() * 1e9);
        
        
        points = {
            "measurement": table_name,
            "tags": tags,
            "fields": fields,
            "time": unixTime
            }

        try:
            self.client.write(points)
        except Exception as e:
            # log error for debugging
            print(f"InfluxDB write error for measurement={table_name}, tags={tags}, time={Measurement_time}: {e}")

    def write_prediction(self, topic, preds, Measurement_time):
        """
        Write prediction fields into the `environment` measurement for the given
        `topic` tag. `preds` is a dict like {'H+1': 12.3, 'H+2': 11.1}.
        This will prefix field names with `pred_` to avoid collision with the
        sensor `value` field.
        """
        if not isinstance(preds, dict):
            return

        fields = {}
        for k, v in preds.items():
            # normalize key to a safe field name (remove '+' and spaces)
            safe_key = k.replace('+', '').replace(' ', '_')
            field_name = f"pred_{safe_key}"
            # convert None to NaN for Influx
            fields[field_name] = (v if v is not None else float('nan'))

        # use the same measurement and tag schema as other writes
        try:
            if self.verbose:
                print(f"write_prediction: topic={topic}, fields={fields}, time={Measurement_time}")
            self.write_data("environment", {"topic": topic}, fields, Measurement_time)
        except Exception as e:
            if self.verbose:
                print(f"Error writing prediction for {topic}: {e}")


    def _write_value_and_predictions(self, topic, last_item):
        """Write the measurement `environment` with the observed `value` plus
        any predictions available from OfflineForecaster for this topic.
        """
        # Build the observed point and include predictions in the same point
        try:
            # Start with observed value
            obs_fields = {"value": last_item['value']}
            # Observed value only. Online predictions are written separately
            # by the OnlineForecaster via `write_prediction()` to keep the
            # responsibilities separated and avoid import cycles.
            self.write_data("environment", {"topic": topic}, obs_fields, last_item['key'])
        except Exception as e:
            if self.verbose:
                print(f"Error writing observed value and predictions for {topic}: {e}")
        
    



    def query_data(self,query):

        '''
        Query : sql query. 
        returns a table.
        '''

        table = self.client.query(query);
        return table;
    





