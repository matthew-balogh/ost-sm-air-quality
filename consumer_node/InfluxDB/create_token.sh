#!/bin/bash

CONTAINER_NAME_OR_ID="$1"
ENV_FILE=".env"

[ -z "$CONTAINER_NAME_OR_ID" ] && exit 1

cat > "$ENV_FILE" << EOF
INFLUX_HOST=http://host.docker.internal:8181
INFLUX_DATABASE=ost-database
EOF

OUTPUT=$(docker exec -i "$CONTAINER_NAME_OR_ID" influxdb3 create token --admin 2>/dev/null | tr -d '\r' | sed 's/\x1b\[[0-9;]*m//g')
TOKEN=$(echo "$OUTPUT" | grep '^Token:' | awk '{print $2}')

[ -n "$TOKEN" ] && echo "INFLUX_TOKEN=$TOKEN" >> "$ENV_FILE"

