# Streamer Monorepo

## How to Run the Streamer

1. **Make the script executable (only needed once):**
   ```sh
   chmod +x start-streaming.sh
   ```

2. **Start the streaming stack:**
   ```sh
   ./start-streaming.sh
   ```
   This will automatically start all services defined in the Docker Compose setup inside the `streamer/` directory.
   - **`org.ost.LocalCsvConsumer` — _Test utility for consuming data in Java. Not used in production; for local testing only._**

## Notes
- Run all development and build commands from the `streamer/` directory unless using the root-level `start-streaming.sh` script.
- Ensure Docker and Docker Compose are installed on your system.
- For PySpark Kafka streaming, use the correct Spark-Kafka connector version matching your Spark installation.
