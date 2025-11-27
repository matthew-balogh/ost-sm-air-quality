FROM python:3.11-slim

WORKDIR /app

# TODO: mount instead of copy ?!

COPY dashboard/dashboard_prototype.py ./dashboard_prototype.py
COPY dashboard/requirements.txt ./requirements.txt

COPY consumer_node/sensor_topics.py ./consumer_node/sensor_topics.py
COPY consumer_node/listeners/sliding_window_listener.py ./consumer_node/listeners/sliding_window_listener.py
COPY consumer_node/anomaly_detector/anomaly_detector.py ./consumer_node/anomaly_detector/anomaly_detector.py
COPY consumer_node/global_statistics/IQR.py ./consumer_node/global_statistics/IQR.py
COPY consumer_node/global_statistics/StreamStatistics.py ./consumer_node/global_statistics/StreamStatistics.py

RUN pip install --no-cache-dir -r requirements.txt

CMD ["streamlit", "run", "dashboard_prototype.py", "--server.address=0.0.0.0"]