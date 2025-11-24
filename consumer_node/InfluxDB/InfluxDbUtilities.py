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

        # instantiate OfflineForecaster singleton so we can fetch predictions
        OfflineForecaster = None
        try:
            # prefer package-style absolute import
            from consumer_node.offline_forcasting.offline_forecasting import OfflineForecaster as _OFF
            OfflineForecaster = _OFF
        except Exception:
            try:
                # fallback: direct module import if running from project root
                from offline_forcasting.offline_forecasting import OfflineForecaster as _OFF
                OfflineForecaster = _OFF
            except Exception:
                # last resort: add project parent to sys.path and try importlib
                try:
                    import importlib, sys
                    parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                    if parent not in sys.path:
                        sys.path.insert(0, parent)
                    mod = importlib.import_module('offline_forcasting.offline_forecasting')
                    OfflineForecaster = getattr(mod, 'OfflineForecaster', None)
                except Exception:
                    OfflineForecaster = None

        try:
            if OfflineForecaster is not None:
                # Use default required_samples (None) so offline forecaster
                # computes required = max(lag_hours) + 1 (typically 25)
                # which matches the original training window.
                self.forecaster = OfflineForecaster(artifacts_dir=None, required_samples=None, verb=verbose)
            else:
                self.forecaster = None
        except Exception:
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



    def on_new_window_co_gt(self, data):
        if self.verbose:
            print("++++++++++++++++++++++++++++++++++++++ INFLUXDB +++++++++++++++++++++++++++++++++");
            print(f"len={len(data)} | " + ", ".join(f"{item['key']}: {item['value']}" for item in data))

        last_item = data[-1]
        self._write_value_and_predictions(topic='co_gt', last_item=last_item)




    def on_new_window_pt08_s1_co(self, data):
        if self.verbose:
            print("++++++++++++++++++++++++++++++++++++++ INFLUXDB +++++++++++++++++++++++++++++++++");
            print(f"len={len(data)} | " + ", ".join(f"{item['key']}: {item['value']}" for item in data))
        last_item = data[-1]
        self._write_value_and_predictions(topic='pt08_s1_co', last_item=last_item)


    def on_new_window_nmhc_gt(self, data):
        if self.verbose:
            print("++++++++++++++++++++++++++++++++++++++ INFLUXDB +++++++++++++++++++++++++++++++++");
            print(f"len={len(data)} | " + ", ".join(f"{item['key']}: {item['value']}" for item in data))
        last_item = data[-1]
        self._write_value_and_predictions(topic='nmhc_gt', last_item=last_item)

    def on_new_window_c6h6_gt(self, data):
        if self.verbose:
            print("++++++++++++++++++++++++++++++++++++++ INFLUXDB +++++++++++++++++++++++++++++++++");
            print(f"len={len(data)} | " + ", ".join(f"{item['key']}: {item['value']}" for item in data))
        last_item = data[-1]
        self._write_value_and_predictions(topic='c6h6_gt', last_item=last_item)

    def on_new_window_pt08_s2_nmhc(self, data):
        if self.verbose:
            print("++++++++++++++++++++++++++++++++++++++ INFLUXDB +++++++++++++++++++++++++++++++++");
            print(f"len={len(data)} | " + ", ".join(f"{item['key']}: {item['value']}" for item in data))
        last_item = data[-1]
        self._write_value_and_predictions(topic='pt08_s2_nmhc', last_item=last_item)

    def on_new_window_nox_gt(self, data):
        if self.verbose:
            print("++++++++++++++++++++++++++++++++++++++ INFLUXDB +++++++++++++++++++++++++++++++++");
            print(f"len={len(data)} | " + ", ".join(f"{item['key']}: {item['value']}" for item in data))
        last_item = data[-1]
        self._write_value_and_predictions(topic='nox_gt', last_item=last_item)

    def on_new_window_pt08_s3_nox(self, data):
        if self.verbose:
            print("++++++++++++++++++++++++++++++++++++++ INFLUXDB +++++++++++++++++++++++++++++++++");
            print(f"len={len(data)} | " + ", ".join(f"{item['key']}: {item['value']}" for item in data))
        last_item = data[-1]
        self._write_value_and_predictions(topic='pt08_s3_nox', last_item=last_item)

    def on_new_window_no2_gt(self, data):
        if self.verbose:
            print("++++++++++++++++++++++++++++++++++++++ INFLUXDB +++++++++++++++++++++++++++++++++");
            print(f"len={len(data)} | " + ", ".join(f"{item['key']}: {item['value']}" for item in data))
        last_item = data[-1]
        self._write_value_and_predictions(topic='no2_gt', last_item=last_item)

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
        
        self.client.write(points);


    def _write_value_and_predictions(self, topic, last_item):
        """Write the measurement `environment` with the observed `value` plus
        any predictions available from OfflineForecaster for this topic.
        """
        fields = {"value": last_item['value']}
        # attempt to get predictions for the corresponding column name
        try:
            if getattr(self, 'forecaster', None) is not None:
                # map topic -> column name used by the forecaster
                topic_to_col = self.forecaster.TOPIC_TO_COLUMN
                target_name = topic_to_col.get(topic)
                if target_name is not None:
                    preds = self.forecaster.predict(target=target_name)
                    if preds:
                        # add prediction fields like pred_H+1, pred_H+2
                        for k, v in preds.items():
                            fields[f"pred_{k}"] = v
        except Exception as e:
            if self.verbose:
                print(f"Error fetching predictions for {topic}: {e}")

        self.write_data("environment", {"topic": topic}, fields, last_item['key'])
    



    def query_data(self,query):

        '''
        Query : sql query. 
        returns a table.
        '''

        table = self.client.query(query);
        return table;
    





