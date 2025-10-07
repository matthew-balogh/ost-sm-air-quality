from pyspark.sql import SparkSession
import os
import sys
from sensor_topics import SENSOR_TOPICS

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
# Initialize SparkSession
spark = SparkSession.builder \
    .appName("KafkaStreamReader") \
    .getOrCreate()

# Set log level to reduce verbosity
spark.sparkContext.setLogLevel("WARN")

# Kafka parameters
kafka_bootstrap_servers = "localhost:9092"  # Change if needed

# Count-based window size (default 8). Can be overridden via env var WINDOW_COUNT
WINDOW_COUNT = int(os.getenv("WINDOW_COUNT", "8"))

# Subscribe to all sensor topics (comma-separated)
topics = ",".join(SENSOR_TOPICS)

# Read from Kafka
kafka_df = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", kafka_bootstrap_servers) \
    .option("subscribe", topics) \
    .load()

# Convert the binary 'key' and 'value' columns to string, and include topic name
messages = kafka_df.selectExpr(
    "CAST(topic AS STRING) as topic",
    "CAST(key AS STRING) as key",
    "CAST(value AS STRING) as value"
)

# Callback function to handle each message
def message_callback(topic, key, value):
    print(f"[CALLBACK] Topic: {topic}, Key: {key}, Value: {value}")

# Callback for when a topic's buffer reaches WINDOW_COUNT messages
def window_callback(topic, records):
    print(f"[WINDOW] Topic: {topic}, Size: {len(records)}")
    for r in records:
        message_callback(r['topic'], r['key'], r['value'])

# Use foreachBatch to process each micro-batch and call the callback for each row
# PySpark does not support a true driver-side callback for each message in a distributed streaming job.
def process_batch(df, epoch_id):
    rows = df.collect()
    # Maintain per-topic buffers across micro-batches
    if not hasattr(process_batch, "buffers"):
        process_batch.buffers = {}
    buffers = process_batch.buffers
    for row in rows:
        topic = row['topic']
        topic_buffer = buffers.setdefault(topic, [])
        topic_buffer.append({
            'topic': row['topic'],
            'key': row['key'],
            'value': row['value']
        })
        # Emit in fixed-size chunks of WINDOW_COUNT
        while len(topic_buffer) >= WINDOW_COUNT:
            chunk = topic_buffer[:WINDOW_COUNT]
            window_callback(topic, chunk)
            del topic_buffer[:WINDOW_COUNT]

query = messages.writeStream \
    .outputMode("append") \
    .foreachBatch(process_batch) \
    .start()

query.awaitTermination()
