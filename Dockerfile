# Use Eclipse Temurin JDK 24 as base
FROM eclipse-temurin:24-jdk

# Set environment variables
ENV KAFKA_VERSION=4.1.0 \
    KAFKA_HOME=/opt/kafka

# Install wget, netcat, and Maven; download and extract Kafka
RUN apt-get update && \
    apt-get install -y wget netcat-traditional maven uuid-runtime && \
    wget https://downloads.apache.org/kafka/${KAFKA_VERSION}/kafka_2.13-${KAFKA_VERSION}.tgz && \
    tar -xzf kafka_2.13-${KAFKA_VERSION}.tgz && \
    mv kafka_2.13-${KAFKA_VERSION} $KAFKA_HOME && \
    rm kafka_2.13-${KAFKA_VERSION}.tgz && \
    if [ ! -f "$KAFKA_HOME/config/kraft/broker.properties" ]; then \
      mkdir -p $KAFKA_HOME/config/kraft && \
      echo "process.roles=broker,controller" > $KAFKA_HOME/config/kraft/broker.properties && \
      echo "node.id=1" >> $KAFKA_HOME/config/kraft/broker.properties && \
      echo "controller.quorum.voters=1@localhost:9093" >> $KAFKA_HOME/config/kraft/broker.properties && \
      echo "listeners=PLAINTEXT://:9092,CONTROLLER://:9093" >> $KAFKA_HOME/config/kraft/broker.properties && \
      echo "controller.listener.names=CONTROLLER" >> $KAFKA_HOME/config/kraft/broker.properties && \
      echo "listener.security.protocol.map=CONTROLLER:PLAINTEXT,PLAINTEXT:PLAINTEXT" >> $KAFKA_HOME/config/kraft/broker.properties && \
      echo "log.dirs=/tmp/kraft-combined-logs" >> $KAFKA_HOME/config/kraft/broker.properties; \
    fi && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the entire project into the container
COPY . /app

# Build the Java app inside the container
RUN mvn clean package -DskipTests

# Copy the startup script and make it executable (already copied above, but ensure permissions)
RUN chmod +x /app/docker-entrypoint.sh

EXPOSE 9092

ENTRYPOINT ["/app/docker-entrypoint.sh"]
