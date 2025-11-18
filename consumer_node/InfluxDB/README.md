# InfluxDB Setup Guide

## Prerequisites

Before starting, ensure you have Docker and Docker Compose installed on your system.
Install the libraries in the **requirements.txt**

## Setup Instructions

### Step 1: Navigate to the InfluxDB Directory

```bash
cd consumer_node/InfluxDB
```

### Step 2: Start InfluxDB with Docker Compose

```bash
docker compose up -d
```

This command will start the InfluxDB container using the configuration specified in `docker-compose.yml`.

### Step 3: Generate Admin Token

Once the container is running, generate an admin token:

```bash
docker exec -it [influxdb3_container_id_or_name] influxdb3 create token --admin
```

**Example output:**
```
apiv3_BP4hoghmK7thvqC8T9tiYdcr48tqLVTGd_36sKhOeivMQ73PlXmN0vjYgyaBFlLajbTNfJRv_6qYknxjtRtPYg
```

> **Note:** Save this token securely - you'll need it for authentication.

OR just use the `create_token.sh` script, it will handle everything.

### Step 4: Access InfluxDB Explorer

Configure a new server connection with the following credentials:

- **Host:** `http://host.docker.internal:8181`
  - Use this address when connecting from another Docker container
- **Admin Token:** Use the token generated in Step 3

## Important Configuration Notes

> ⚠️ **Warning:** Remember to update your `.env` file with the generated admin token and any other necessary configuration values before proceeding.

## Troubleshooting

- To find your container ID or name, run: `docker ps`
- If you're connecting from the host machine (not another container), use `http://localhost:8181` instead
- Ensure no other services are using port 8181 before starting the container
- If you get **Token already created** message, delete : 
`\.influxdb3\core\data`
&
`\.influxdb3\core\plugins`
