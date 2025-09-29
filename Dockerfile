# Use Eclipse Temurin JDK 24 as base
FROM eclipse-temurin:24-jdk

WORKDIR /app

# Install Maven
RUN apt-get update && apt-get install -y maven && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy the entire project into the container
COPY . /app

# Build the Java app inside the container
RUN mvn clean package -DskipTests

CMD ["java", "-jar", "target/sensor_data_streamer-1.0-SNAPSHOT-jar-with-dependencies.jar"]
