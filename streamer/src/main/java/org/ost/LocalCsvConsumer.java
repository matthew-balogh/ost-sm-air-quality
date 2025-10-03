package org.ost;

import org.apache.kafka.clients.consumer.ConsumerConfig;
import org.apache.kafka.clients.consumer.ConsumerRecords;
import org.apache.kafka.clients.consumer.KafkaConsumer;
import org.apache.kafka.clients.consumer.ConsumerRecord;
import org.apache.kafka.common.serialization.StringDeserializer;

import java.time.Duration;
import java.util.Arrays;
import java.util.Properties;
import java.util.UUID;

import static org.ost.Main.SENSOR_TOPICS;

public class LocalCsvConsumer {
    

    public static void main(String[] args) {
        // Use the same bootstrap server as Main
        String bootstrapServers = System.getenv().getOrDefault("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092");
        String groupId = System.getenv().getOrDefault("KAFKA_CONSUMER_GROUP", "sensor-consumer-group-" + UUID.randomUUID());

        Properties props = new Properties();
        props.put(ConsumerConfig.BOOTSTRAP_SERVERS_CONFIG, bootstrapServers);
        props.put(ConsumerConfig.GROUP_ID_CONFIG, groupId);
        props.put(ConsumerConfig.KEY_DESERIALIZER_CLASS_CONFIG, StringDeserializer.class.getName());
        props.put(ConsumerConfig.VALUE_DESERIALIZER_CLASS_CONFIG, StringDeserializer.class.getName());
        props.put(ConsumerConfig.AUTO_OFFSET_RESET_CONFIG, "earliest");
        props.put(ConsumerConfig.ENABLE_AUTO_COMMIT_CONFIG, "true");

        try (KafkaConsumer<String, String> consumer = new KafkaConsumer<>(props)) {
            // Subscribe to all sensor topics
            consumer.subscribe(Arrays.asList(SENSOR_TOPICS));
            System.out.println("Subscribed to sensor topics: " + Arrays.toString(SENSOR_TOPICS));
            System.out.println("Consumer group: " + groupId);
            System.out.println("Listening for messages...\n");
            
            while (true) {
                ConsumerRecords<String, String> records = consumer.poll(Duration.ofMillis(1000));
                for (ConsumerRecord<String, String> record : records) {
                    System.out.printf("[%s] Key: %s, Value: %s, Partition: %d, Offset: %d%n",
                            record.topic(), record.key(), record.value(), record.partition(), record.offset());
                }
            }
        }
    }
}
