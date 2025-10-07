# Sensor Data Streamer - PySpark Kafka Consumer

This project demonstrates how to consume multiple Kafka topics using PySpark Structured Streaming. It prints the topic name, key, and value for each message received from the specified sensor topics.

## Prerequisites

- Python 3.11
- Apache Spark 4.0.1 (pre-built for Scala 2.13)
- Kafka broker running and accessible (default: `localhost:9092`)
- JAVA 21 or higher
- 
```
How to run???
spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.13:3.5.1 read_kafka_stream.py
```=