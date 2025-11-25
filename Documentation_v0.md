# Open-source Technologies and Stream Mining joint Project Documentation

## I. Project details

**Team members:**
* Albazzal, Houmam
* Balogh, Máté
* Földvári, Ádám
* Lahmar, Abderraouf
* Nagy, Zsuzsanna

**Mentored by:**
Loubna Seddiki

**Topic:**
Smart City Air Quality Monitoring with Real-Time Stream Analytics (SCAir-IoT)

**Description:**
This project will develop a system to collect, store, and analyze air quality sensor data from smart city IoT devices. The system will simulate multiple sensors measuring pollutants (PM2.5, CO₂, NO₂, O₃) and detect pollution spikes in real time. It will also forecast short-term air quality using batch analytics to support city planning and public health responses.

**Dataset:**
* UCI Air Quality Dataset 
* Optionally enriched with open city datasets

## II. Planning

**Outline of project:**
We are planning to split the time-ordered dataset at a certain data and consider the first part as a static historical data used for training a global machine learning model used for forecasting and anomaly detection. The second part is simulated as a continuous stream data wich is processed in mini-batches (windows) in real time. The in-window data is used directly to make predictions such as up-to-date forecasts and detection of spikes in the sensor measurements, and only then is saved to the database for further use. To avoid the global model become outdated, we periodically (after one month of new data) re-train the model. A dashboard is created to display the sensor measurements along with the forecasted values making them easily comparable by the human eye. Alert widgets will be included in the dashboard indicating where (and how) the anomalies occurred. Optionally, we setup an alerting mechanism that sends out external notification on critical anomalies.

**Objectives:**
* Architecture with loosely connected self-contained elements (Docker containers)
* Train a first static, later periodically updated ML model for a global opinion on forecasting and anomaly detection
* Process streamlined data, identify anomalies and forecast 3 hours ahead using the pre=learned global information and the real-time (window) data
* Persist incoming data in database
* Live dashboard
* External notifications (optional)

### Architecture diagram

![](./_images/architecture_design.png)

#### Raw data

Static file containing sensor measurements related to air quality.

#### Sensor data simulator

Reads the raw data file and simulates flow of sensor data.

#### Kafka producer

Streamlines the simulated sensor data into Kafka topics in their datetime order.

#### PySpark consumer

Listens to the streamlined data and creates mini batches to call analytical functions—such as Anomaly Detection and Forecasting—on this windowed data.

#### Influxdb

Data is then persisted in the database including the online predictions and the original incoming data.

#### Streamlit

Streamlit is used to create dashboard prototypes without the necessity of a database storage and connection. Kafka messages are directly consumed by this component to display simple line charts and anomaly detection related information and alerts.

#### Grafana

Dashboard visualization component that periodically fetches the database for new data to show the latest insights in real-time.

### Modeling diagram

![](./_images/modeling_design.png)

#### Static (offline) zone

In this zone, **static models** are trained and evaluated. These models are then utilized (but not updated) in the *Online zone* as baseline predictors.

#### Real-time (online) zone

In this zone, the raw data is streamlined to simulate real-time data flow. This flow is continuously processed and **models** are created and maintained through **incremental updates** to make better predictions in forecasting and anomaly detection.


### Team member responsibilities

| Team member        | Main responsibilities                               |
| ------------------ | --------------------------------------------------- |
| Albazzal, Houmam   | Containerization, Data Streamlining                 |
| Balogh, Máté       | Anomaly Detection, Pipeline Integration             |
| Földvári, Ádám     | ML modeling, Forecasting                            |
| Lahmar, Abderraouf | Calculation and Combination of Statistics, Database |
| Nagy, Zsuzsanna    | Dashboarding                                        |

#### Containerization

Creating standalone self-contained Docker containers for the elements depicted in the architecture diagram.

#### Data streamlining

Turning raw data into an ordered data flow, organized into Kafka topics.

#### Anomaly detection

Detecting sudden changes in sensor readings.

#### Pipeline integration

Overseeing and ensuring all the standalone components are compatible with each other.

#### ML modeling, forecasting

Preparing the incoming data, selecting applicable ML algorithms and making forecasts on future sensor readings, along with classification of air quality.

#### Statistics

Ensuring statistical functions are available at the online analytics phase, along with maintaining global estimations if necessary.

#### Database

Creating database schema, along with setting up connections to persist and to read data.

#### Dashboarding

Creating visualizations for forecasted sensor measurements and air quality classification, along with anomalies and the actual sensor readings.

## III. Background and Literature Review

### Stream mining

In stream mining, we are limited to a portion of data and make decisions real-time in memory. In traditional machine learning contexts, data is referred to as batch data which can be loaded into memory in its entirety [1]. According to the authors, "this is of stark contrast to stream mining, where data streams produce elements in a sequential, continuous fashion, and may also be impermanent, or transient, in nature ... This means stream data may only be available for a short time." The authors refer to [2], highlighting that "once an element from a data stream has been processed, it is discarded or archived. It cannot be retrieved easily unless it is explicitly stored in memory, which is small relative to the size of data streams."


### References

[1]  Wares, S., Isaacs, J. and Elyan, E. (2019). Data stream mining: methods and challenges for handling concept drift. \textit{SN Applied Sciences}, 1(11), 1412.

[2]  Babcock, B., Babu, S., Datar, M., Motwani, R. and Widom, J. (2002, June). Models and issues in data stream systems. In \textit{Proceedings of the twenty-first ACM SIGMOD-SIGACT-SIGART symposium on Principles of database systems} (pp. 1–16).