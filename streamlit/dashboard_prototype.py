import sys, os
import json, time
import numpy as np
import pandas as pd
import streamlit as st

consumer_node_dir = os.path.abspath("consumer_node")
sys.path.insert(1, consumer_node_dir)

from sensor_topics import SENSOR_TOPICS
from anomaly_detector.anomaly_detector import InWindowAnomalyDetector
from confluent_kafka import Consumer, KafkaError

# FIXME: preprocessing is done here (-200 to np.nan)

BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "broker:29092")
GROUP_ID = "sensor-dashboard"
BUFFER_SIZE = 8
LOG_SIZE = 10

st.set_page_config(page_title="Air Quality Live Dashboard", layout="wide")

def on_topic_selected():
    selected_topic = st.session_state.selectedTopic
    st.toast(f"Sensor for \"{selected_topic}\" is selected")

    if selected_topic != SENSOR_TOPICS[0]:
        st.toast(f"This sensor is currently not available", icon=":material/warning:")

with st.container(horizontal=True, vertical_alignment="bottom"):
    with st.container():
        st.title("Air Quality Live Dashboard")
        st.caption(f"This dashboard displays :blue[**air quality**] measurements of an :red[**Italian city**] with respect to the **selected sensor**.")

    topic = st.selectbox("Sensor", options=SENSOR_TOPICS, on_change=on_topic_selected, key="selectedTopic", width=250)


conf = {
    'bootstrap.servers': BOOTSTRAP,
    'group.id': GROUP_ID,
    'auto.offset.reset': 'latest'
}

consumer = Consumer(conf)
consumer.subscribe([topic])

readings = pd.Series()
detector = InWindowAnomalyDetector()

df_log = pd.DataFrame(columns=['sensor', 'value', 'abnormal', 'abnormality_type', 'abnormality_reason'])
df_anomaly_counts = pd.Series(np.zeros(3, dtype=int), index=["global", "local", "missing"])

st.info("Listening for Kafka messages…")

col1, col2 = st.columns([4, 2], gap="large")

with col1:
    st.header("Sensor readings")
    placeholder_chart = st.empty()

with col2:
    st.header("Abnormal detections")
    placeholder_metrics = st.empty()

try:
    while True:
        msg = consumer.poll(1.0)

        if msg is None:
            continue
        if msg.error():
            if msg.error().code() != KafkaError._PARTITION_EOF:
                st.error(str(msg.error()))
            continue

        try:
            reading = json.loads(msg.value().decode("utf-8"))
            print("Consumed:", reading)
            reading = np.nan if reading == -200 else reading
            readings.loc[len(readings)] = reading
        except Exception:
            continue

        sr_buffer = readings[-BUFFER_SIZE:]

        with placeholder_chart.container():
            if not sr_buffer.empty:
                reading = sr_buffer.iloc[-1]

                with st.container(horizontal=True):
                    st.subheader(topic)
                    with st.container(horizontal=True, horizontal_alignment="right"):
                        st.metric(topic, round(reading, 2), label_visibility="hidden", width="content")

                if hasattr(detector, f'on_new_window_{topic}'):
                    is_anomalous, predictions = getattr(detector, f'on_new_window_{topic}')(
                        [{
                            "topic": topic,
                            "key": "no_key",
                            "value": x
                        } for x in sr_buffer]
                    )
                    
                st.line_chart(sr_buffer, x_label="Time", y_label="Sensor value", use_container_width=True, height=260)

                is_anomalous = any(p["pred"] for p in predictions)
                types = [p["type"] for p in predictions if p["pred"]]
                reasons = [p["message"] for p in predictions if p["pred"]]

                for pred in predictions:
                    if pred["pred"]:
                        df_anomaly_counts[pred['type']] += 1

                obs = [
                    topic,
                    reading,
                    is_anomalous,
                    types if is_anomalous else np.nan,
                    reasons if is_anomalous else np.nan
                ]

                df_log.loc[len(df_log)] = obs

                st.caption("Log of recent readings:")
                st.dataframe(df_log.sort_index(ascending=False)[:10])
            
        with placeholder_metrics.container():
            st.metric("Missing value", df_anomaly_counts['missing'])
            st.metric("Global outlier", df_anomaly_counts['global'])
            st.metric("Local outlier", df_anomaly_counts['local'])

            st.caption("Log of recent detections:")
            for t, r in df_log[df_log['abnormal']].sort_index(ascending=False)[:10].iterrows():
                st.error(f"At T={t} anomaly was detected in \"{topic} sensor\" due to {r['abnormality_reason']}", icon="🚨")

        time.sleep(1.5)

finally:
    consumer.close()
