Dataset can be accessed here: https://archive.ics.uci.edu/dataset/360/air+quality

AirQualityUCI.csv, put it into a Data folder within the root folder.

Run the .py file:
spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.13:3.5.1 read_kafka_stream.py