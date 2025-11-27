from pyspark.sql import SparkSession
import os
import sys

from offline_forcasting.offline_forecasting import OfflineForecaster
from sensor_topics import SENSOR_TOPICS
from InfluxDB import InfluxDbUtilities
from trend_detector.TrendDetector import InWindowMKTrendDetector
from global_statistics.StreamStatistics import SimpleTDigest

from anomaly_detector.anomaly_detector import InWindowAnomalyDetector
from anomaly_detector.novelty_function import derivateNoveltyFn
from anomaly_detector.outlier_estimator import MissingValueDetector, WindowOutlierDetector, TDigestOutlierDetector

class KafkaStreamReader:
    
    def __init__(self):

        self.observers = []
        os.environ['PYSPARK_PYTHON'] = sys.executable
        os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

        self.spark = SparkSession.builder \
            .appName("KafkaStreamReader") \
            .getOrCreate()
        self.spark.sparkContext.setLogLevel("WARN")
        self.kafka_bootstrap_servers = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
        self.WINDOW_COUNT = int(os.getenv("WINDOW_COUNT", "8"))
        self.topics = ",".join(SENSOR_TOPICS)
        self.kafka_df = self.spark.readStream \
            .format("kafka") \
            .option("kafka.bootstrap.servers", self.kafka_bootstrap_servers) \
            .option("subscribe", self.topics) \
            .load()
        self.messages = self.kafka_df.selectExpr(
            "CAST(topic AS STRING) as topic",
            "CAST(key AS STRING) as key",
            "CAST(value AS STRING) as value"
        )

    def window_callback(self, topic, records):
        # Notify all observers for this topic
        method_name = f'on_new_window_{topic}'
        for observer in self.observers:
            if hasattr(observer, method_name):
                getattr(observer, method_name)(records)

    def process_batch(self, df, epoch_id):
        rows = df.collect()
        if not hasattr(self, "buffers"):
            self.buffers = {}
        buffers = self.buffers
        for row in rows:
            topic = row['topic']
            topic_buffer = buffers.setdefault(topic, [])
            topic_buffer.append({
                'topic': row['topic'],
                'key': row['key'],
                'value': row['value']
            })
            if len(topic_buffer) > self.WINDOW_COUNT:
                topic_buffer.pop(0)  # Remove the oldest record
            # Always invoke callback with the current window (sliding)
            self.window_callback(topic, list(topic_buffer))

    def run(self):
        query = self.messages.writeStream \
            .outputMode("append") \
            .foreachBatch(self.process_batch) \
            .start()
        query.awaitTermination()

    def register_observer(self, observer):
        if observer not in self.observers:
            self.observers.append(observer)

    def remove_observer(self, observer):
        if observer in self.observers:
            self.observers.remove(observer)

if __name__ == "__main__":

    reader = KafkaStreamReader()

    databaseWriter = InfluxDbUtilities.DatabaseWriter(verbose=True);
    ## To fix Grafana problenms with missing measurements, create all measurements at the start
    databaseWriter.create_all_measurements();

    reader.register_observer(databaseWriter)

    anomalyDetector = InWindowAnomalyDetector(
        dbWriter=databaseWriter,
        novelty_fn=derivateNoveltyFn,
        estimators={
            "global": TDigestOutlierDetector(tdigest=SimpleTDigest(delta=.1), upper_only=True),
            "local": WindowOutlierDetector(upper_only=True),
            "missing":  MissingValueDetector(),
        },
        verb=True)
    
    reader.register_observer(anomalyDetector)

    forecaster = OfflineForecaster(verb=True)
    reader.register_observer(forecaster)

    Trend_detector = InWindowMKTrendDetector(verbose=True, t_digest_compression_delta=0.06, quantile_step = 5, dbWriter = databaseWriter);
    reader.register_observer(Trend_detector);


    reader.run()
