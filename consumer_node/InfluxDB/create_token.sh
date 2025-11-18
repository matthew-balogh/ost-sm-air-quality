#!/bin/bash

# --- CONFIG ---
CONTAINER_NAME_OR_ID="$1"    # passed as the first argument
ENV_FILE=".env"
ENV_VAR="INFLUX_TOKEN"

if [ -z "$CONTAINER_NAME_OR_ID" ]; then
  echo "Usage: ./create_token.sh <influxdb3_container_name_or_id>"
  exit 1
fi

# --- CREATE TOKEN IN CONTAINER ---
echo "Creating InfluxDB admin token in container: $CONTAINER_NAME_OR_ID"
OUTPUT=$(docker exec -it "$CONTAINER_NAME_OR_ID" influxdb3 create token --admin | tr -d '\r')

# remove ANSI escape characters (from -it)
OUTPUT=$(echo "$OUTPUT" | sed 's/\x1b\[[0-9;]*m//g')

# --- EXTRACT TOKEN ---
TOKEN=$(echo "$OUTPUT" | grep '^Token:' | awk '{print $2}')

if [ -z "$TOKEN" ]; then
  echo "No token found in output. Nothing added to $ENV_FILE."
  exit 0
fi

echo "Generated token: $TOKEN"

# --- WRITE TO .env ON A NEW LINE ---
# Ensure file ends with newline
touch "$ENV_FILE"
if [ -n "$(tail -c1 "$ENV_FILE")" ]; then
    echo "" >> "$ENV_FILE"
fi

# If variable exists → replace it; else → append on a new line
if grep -q "^$ENV_VAR=" "$ENV_FILE"; then
  sed -i "s/^$ENV_VAR=.*/$ENV_VAR=$TOKEN/" "$ENV_FILE"
else
  echo "$ENV_VAR=$TOKEN" >> "$ENV_FILE"
fi

echo "Token saved to $ENV_FILE under $ENV_VAR"
