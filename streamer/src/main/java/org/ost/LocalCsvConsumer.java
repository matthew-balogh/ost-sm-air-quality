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

public class LocalCsvConsumer {
    
    // List of sensor topics (same as in Main.java)
    private static final String[] SENSOR_TOPICS = {
        "co_gt",           // CO(GT)
        "pt08_s1_co",      // PT08.S1(CO)
        "nmhc_gt",         // NMHC(GT)
        "c6h6_gt",         // C6H6(GT)
        "pt08_s2_nmhc",    // PT08.S2(NMHC)
        "nox_gt",          // NOx(GT)
        "pt08_s3_nox",     // PT08.S3(NOx)
        "no2_gt",          // NO2(GT)
        "pt08_s4_no2",     // PT08.S4(NO2)
        "pt08_s5_o3",      // PT08.S5(O3)
        "t",               // T
        "rh",              // RH
        "ah"               // AH
    };
    
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
