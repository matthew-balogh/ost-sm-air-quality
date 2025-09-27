#!/bin/bash
set -e

# Set up KRaft storage directory if not already initialized
if [ ! -d "/tmp/kraft-combined-logs" ]; then
  "$KAFKA_HOME/bin/kafka-storage.sh" format -t $(uuidgen) -c "$KAFKA_HOME/config/kraft/broker.properties"
fi
#
# Start Kafka in KRaft mode (no Zookeeper)
"$KAFKA_HOME/bin/kafka-server-start.sh" "$KAFKA_HOME/config/kraft/broker.properties" & KAFKA_PID=$!

# Wait for Kafka to be ready
while ! nc -z localhost 9092; do
  echo "Waiting for Kafka broker to start..."
  sleep 1
done

# Create the Kafka topic if it doesn't exist
$KAFKA_HOME/bin/kafka-topics.sh --create --if-not-exists --topic sensor_data --bootstrap-server localhost:9092
$KAFKA_HOME/bin/kafka-topics.sh --describe --topic sensor_data --bootstrap-server localhost:9092

# Run the Java application
java -jar /app/target/sensor_data_streamer-1.0-SNAPSHOT-jar-with-dependencies.jar

# Wait for Kafka process to end (if Java app exits)
wait $KAFKA_PID
