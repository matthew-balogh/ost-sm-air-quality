package org.ost;

//TIP To <b>Run</b> code, press <shortcut actionId="Run"/> or
// click the <icon src="AllIcons.Actions.Execute"/> icon in the gutter.
import org.apache.kafka.clients.producer.KafkaProducer;
import org.apache.kafka.clients.producer.ProducerRecord;
import org.apache.kafka.clients.producer.RecordMetadata;
import org.apache.kafka.clients.producer.ProducerConfig;
import org.apache.kafka.common.serialization.StringSerializer;

import java.util.Properties;

public class Main {
    static String topic = "sensor_data";
    public static void main(String[] args) {
        String bootstrapServers = System.getenv().getOrDefault("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092");
        Properties props = new Properties();
        props.put(ProducerConfig.BOOTSTRAP_SERVERS_CONFIG, bootstrapServers);
        props.put(ProducerConfig.KEY_SERIALIZER_CLASS_CONFIG, StringSerializer.class.getName());
        props.put(ProducerConfig.VALUE_SERIALIZER_CLASS_CONFIG, StringSerializer.class.getName());

        try (KafkaProducer<String, String> producer = new KafkaProducer<>(props)) {
            String message = "Hello from sensor_data_streamer!";
            ProducerRecord<String, String> record = new ProducerRecord<>(topic, "key1", message);
            RecordMetadata metadata = producer.send(record).get();
            System.out.printf("Sent message to topic %s partition %d offset %d\n", metadata.topic(), metadata.partition(), metadata.offset());
        } catch (Exception e) {
            System.err.println("Failed to send message to Kafka: " + e.getMessage());
            e.printStackTrace();
        }
    }
}