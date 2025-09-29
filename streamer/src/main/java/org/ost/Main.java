package org.ost;

//TIP To <b>Run</b> code, press <shortcut actionId="Run"/> or
// click the <icon src="AllIcons.Actions.Execute"/> icon in the gutter.
import org.apache.kafka.clients.producer.*;
import org.apache.kafka.common.serialization.StringSerializer;
import org.apache.kafka.clients.admin.*;

import java.util.*;
import java.util.concurrent.ExecutionException;
import java.util.Properties;
import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.InputStream;

public class Main {
    static String topic = "sensor-data";
    public static void main(String[] args) {
        int skipLines = 3000; // Number of lines to skip (e.g., header)
        // Read bootstrap servers from environment variable, fallback to localhost:9092
        final String bootstrapServers = System.getenv().getOrDefault("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092");
        Properties props = new Properties() {{
            put(ProducerConfig.BOOTSTRAP_SERVERS_CONFIG, bootstrapServers);
            put(ProducerConfig.KEY_SERIALIZER_CLASS_CONFIG, StringSerializer.class);
            put(ProducerConfig.VALUE_SERIALIZER_CLASS_CONFIG, StringSerializer.class);
            put(ProducerConfig.ACKS_CONFIG, "1");
        }};

//        Producer<String, String> producer = new KafkaProducer<>(props);


        final String key = "key1";
        final String value = "value1";
        ProducerRecord<String, String> record = new ProducerRecord<>(topic, key, value);

//        producer.send(record);
//        producer.flush();
//        producer.close();
        System.out.println("Message sent to topic '" + topic + "' with key '" + key + "' and value '" + value + "'.");

        // Uncomment below to stream CSV data to Kafka

        try (KafkaProducer<String, String> producer = new KafkaProducer<>(props)) {
            // Read CSV from resources
            InputStream is = Main.class.getClassLoader().getResourceAsStream("AirQualityUCI.csv");
            if (is == null) {
                System.err.println("Could not find AirQualityUCI.csv in resources.");
                return;
            }
            BufferedReader reader = new BufferedReader(new InputStreamReader(is));
            String line;
            int rowNum = 0;
            // Skip the first 'skipLines' lines
            for (int i = 0; i < skipLines; i++) {
                reader.readLine();
            }
            while ((line = reader.readLine()) != null) {
                rowNum++;
                ProducerRecord<String, String> csvRecord = new ProducerRecord<>(topic, Integer.toString(rowNum), line);
                RecordMetadata metadata = producer.send(csvRecord).get();
                System.out.printf("Sent row %d to topic %s partition %d offset %d\n", rowNum, metadata.topic(), metadata.partition(), metadata.offset());
                producer.flush();
                Thread.sleep(1000); // 1 second
            }
            reader.close();
            producer.close();
        } catch (Exception e) {
            System.err.println("Failed to stream CSV to Kafka: " + e.getMessage());
            e.printStackTrace();
        }

    }
}