# Use Eclipse Temurin OpenJDK 21 base image
FROM eclipse-temurin:21-jdk

# Install Python 3.10 and system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get update && \
    apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3.10-distutils \
    && rm -rf /var/lib/apt/lists/*

# Install pip for Python 3.10
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10 && \
    ln -sf /usr/bin/python3.10 /usr/local/bin/python3 && \
    ln -sf /usr/bin/python3.10 /usr/local/bin/python && \
    python3.10 -m pip install --upgrade pip

# Set environment variables for Spark
ENV SPARK_VERSION=4.0.1
ENV HADOOP_VERSION=3
ENV SPARK_HOME=/opt/spark
ENV PYTHONUNBUFFERED=1
ENV PYSPARK_PYTHON=python3.10
ENV PYSPARK_DRIVER_PYTHON=python3.10
ENV JAVA_HOME=/opt/java/openjdk
ENV PATH=$PATH:${SPARK_HOME}/bin:${SPARK_HOME}/sbin:/usr/local/bin

# Install Spark 4.0.1
RUN wget -q https://archive.apache.org/dist/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz && \
    tar -xzf spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz -C /opt && \
    mv /opt/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION} ${SPARK_HOME} && \
    rm spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN python3.10 -m pip install --no-cache-dir --break-system-packages -r requirements.txt

# Copy the entire consumer_node directory
COPY . .

# Make run script executable (if needed)
RUN chmod +x run_spark.sh || true

# Default command - run spark-submit with Kafka package
CMD ["spark-submit", "--packages", "org.apache.spark:spark-sql-kafka-0-10_2.13:3.5.1", "read_kafka_stream.py"]
