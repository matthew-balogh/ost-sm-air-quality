#!/bin/bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
STREAMER_DIR="${PROJECT_ROOT}/streamer"
ENV_FILE="${PROJECT_ROOT}/consumer_node/InfluxDB/.env"
GENERATE_TOKEN_SCRIPT="${PROJECT_ROOT}/consumer_node/InfluxDB/generate_token.sh"


cd "${STREAMER_DIR}"

echo "Starting InfluxDB services..."


## I did not understand this part
docker compose -f docker-compose-influx.yml up -d
# docker compose up -d influxdb3-core influxdb3-explorer

echo "InfluxDB is ready."


echo "Ensuring INFLUX_HOST is set in ${ENV_FILE}..."
INFLUX_HOST_ENTRY="INFLUX_HOST=http://influxdb3-core:8181"
if [ -f "${ENV_FILE}" ]; then
  if grep -q "^INFLUX_HOST=" "${ENV_FILE}"; then
    if [[ "$OSTYPE" == "darwin"* ]]; then
      sed -i '' "s|^INFLUX_HOST=.*|${INFLUX_HOST_ENTRY}|" "${ENV_FILE}"
    else
      sed -i "s|^INFLUX_HOST=.*|${INFLUX_HOST_ENTRY}|" "${ENV_FILE}"
    fi
  else
    echo "${INFLUX_HOST_ENTRY}" >> "${ENV_FILE}"
  fi
else
  printf "%s\n" "${INFLUX_HOST_ENTRY}" > "${ENV_FILE}"
fi

INFLUX_CONTAINER_ID=$(docker compose ps -q influxdb3-core)

if [ -z "${INFLUX_CONTAINER_ID}" ]; then
  echo "Error: Could not determine influxdb3-core container ID."
  exit 1
fi

echo "Generating InfluxDB token..."
(
  cd "${PROJECT_ROOT}/consumer_node/InfluxDB"
  ./create_token.sh "${INFLUX_CONTAINER_ID}"
)

 echo "Starting full stack..."
 exec docker compose up --build -d

