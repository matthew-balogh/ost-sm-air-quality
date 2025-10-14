from pyspark.sql import SparkSession
import os
import sys
from sensor_topics import SENSOR_TOPICS
from anomaly_detector.anomaly_detector import InWindowAnomalyDetector
from global_statistics.StreamStatistics import SimpleTDigest

class KafkaStreamReader:
    def __init__(self):
        self.observers = []
        os.environ['PYSPARK_PYTHON'] = sys.executable
        os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

        self.spark = SparkSession.builder \
            .appName("KafkaStreamReader") \
            .getOrCreate()
        self.spark.sparkContext.setLogLevel("WARN")
        self.kafka_bootstrap_servers = "localhost:9092"
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
    
    globalStatistics = SimpleTDigest(delta=0.1) # for CO only, FIXME later
    detector = InWindowAnomalyDetector(globalStatistics=globalStatistics, verb=False)

    reader.register_observer(detector)
    reader.run()
