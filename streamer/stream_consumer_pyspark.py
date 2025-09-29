# NOTE: For best compatibility, use Java 8 or 11 with PySpark.
# If you have multiple Java versions, set JAVA_HOME to Java 8 or 11 before running this script.

from pyspark.sql import SparkSession
from pyspark.sql.functions import col

# Create Spark session with Ka fka su   pport
def main():
    spark = SparkSession.builder \
        .appName("KafkaCsvStreamConsumer") \
        .master("local[*]") \
        .getOrCreate()

    # Set log level to WARN to reduce noise
    spark.sparkContext.setLogLevel("WARN")

    # Read from Kafka topic 'sensor_data' on broker:29092 (Docker network)
    df = spark.readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", "broker:29092") \
        .option("subscribe", "sensor-data") \
        .option("startingOffsets", "earliest") \
        .load()

    # The value is in binary, so cast to string
    csv_lines = df.selectExpr("CAST(value AS STRING)")

    # Print each row as it arrives
    query = csv_lines.writeStream \
        .outputMode("append") \
        .format("console") \
        .option("truncate", False) \
        .start()

    query.awaitTermination()

if __name__ == "__main__":
    main()
