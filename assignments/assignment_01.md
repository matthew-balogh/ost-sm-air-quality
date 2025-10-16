# Assignment 1

System architecture diagram and team member responsibilites of **Team T5_AirQual_Mate**.

## Architecture design

![](../resources/architecture_design.png)

### Raw data

Static file containing sensor measurements related to air quality.

### Kafka producer

Reads the raw data file and streamlines the records into Kafka topics in their datetime order.

### PySpark consumer

Listens to the streamlined data and creates mini batches to call analytical functions—such as Anomaly Detection and Forecasting—on this windowed data.

### Influxdb

Data is then persisted in the database including the online predictions and the original incoming data.

### Grafana

Dashboard visualization component that periodically fetches the database for new data to show the latest insights in real-time.
