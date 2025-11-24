#!/bin/bash

# Kafka bootstrap server
BOOTSTRAP_SERVER="localhost:9092"

# List of topics
topics=(
  "co_gt"
  "pt08_s1_co"
  "nmhc_gt"
  "c6h6_gt"
  "pt08_s2_nmhc"
  "nox_gt"
  "pt08_s3_nox"
  "no2_gt"
  "pt08_s4_no2"
  "pt08_s5_o3"
  "t"
  "rh"
  "ah"
)

for topic in "${topics[@]}"
do
  echo "Creating topic: $topic"
  kafka-topics.sh --create \
    --bootstrap-server "$BOOTSTRAP_SERVER" \
    --replication-factor 1 \
    --partitions 1 \
    --topic "$topic"
done

echo "All topics created."