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

# Use foreachBatch to process each micro-batch and call the callback for each row
def process_batch(df, epoch_id):
    rows = df.collect()
    for row in rows:
        message_callback(row['topic'], row['key'], row['value'])

query = messages.writeStream \
    .outputMode("append") \
    .foreachBatch(process_batch) \
    .start()

query.awaitTermination()
