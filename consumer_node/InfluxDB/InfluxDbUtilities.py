## We will use Python client API to write, fetch data
## Install, "pip install influxdb3-python pandas"

import os

from dotenv import load_dotenv
from datetime import datetime
from influxdb_client_3 import (InfluxDBClient3, InfluxDBError, Point, WriteOptions, write_client_options)
from listeners.sliding_window_listener import SlidingWindowListener


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
    

    def write_trend(self, sample, type, topic,S_value):
        '''
        data should have the same structure as other stream data + trend type + topic name (e.g: co_gt,...).

        '''
        self.write_data("trend", {"topic": topic},
                        {
                            "value": sample['value'],
                            "S": S_value,
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


    def write_forecasting_data(self, topic, preds,key):
        """
        Write forecasting predictions into InfluxDB for the given topic.
        `preds` is a dict like {'H+1': 12.3, 'H+2': 11.1}.
        """
        try:
            if self.verbose:
                print(f"write_forecasting_data: topic={topic}, preds={preds}, time={key}")
            # self.write_prediction(topic, preds, measurement_time)
            self.write_data("forecasting", {"topic": topic}, preds,key);
        
        except Exception as e:
            if self.verbose:
                print(f"Error writing forecasting data for {topic}: {e}")


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

            # attempt to get predictions for the corresponding column name
            preds = None
            if getattr(self, 'forecaster', None) is not None:
                topic_to_col = self.forecaster.TOPIC_TO_COLUMN
                target_name = topic_to_col.get(topic)
                if target_name is not None:
                    try:
                        preds = self.forecaster.predict(target=target_name)
                    except Exception as e:
                        if self.verbose:
                            print(f"Error calling forecaster.predict for {target_name}: {e}")

                    if self.verbose:
                        print(f"DatabaseWriter: predict() returned for {target_name}: {preds}")

                    # If predict() returned nothing, fall back to most recent periodic predictions
                    if not preds:
                        try:
                            last = getattr(self.forecaster, 'last_predictions', None)
                            if last is not None and target_name in last:
                                preds = last.get(target_name)
                                if self.verbose:
                                    print(f"DatabaseWriter: using last_predictions for {target_name}: {preds}")
                        except Exception as e:
                            if self.verbose:
                                print(f"Error accessing last_predictions for {target_name}: {e}")

            # If we have predictions, attach them to the observed point (temporary UI-friendly behavior)
            if preds:
                for k, v in preds.items():
                    # sanitize key (e.g. 'H+1' -> 'H1')
                    safe_k = k.replace('+', '')
                    obs_fields[f"pred_{safe_k}"] = v

            # Write single combined point (value + predictions)
            self.write_data("environment", {"topic": topic}, obs_fields, last_item['key'])
        except Exception as e:
            if self.verbose:
                print(f"Error writing observed value and predictions for {topic}: {e}")
        except Exception as e:
            if self.verbose:
                print(f"Error fetching/writing predictions for {topic}: {e}")


    def query_data(self,query):
        '''
        Query : sql query. 
        returns a table.
        '''

        table = self.client.query(query);
        return table;
