package org.ost;
import org.apache.kafka.clients.producer.*;
import org.apache.kafka.common.serialization.StringSerializer;
import org.apache.kafka.clients.admin.*;
import org.apache.kafka.common.errors.TopicExistsException;

import java.util.*;
import java.util.concurrent.ExecutionException;
import java.util.Properties;
import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.InputStream;

public class Main {
    static String topic = "sensor-data";
    
    // List of topics based on CSV columns (excluding Date, Time, and empty columns)
    public static final String[] SENSOR_TOPICS = {
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
    
    // CSV column indices (0-based, excluding Date=0, Time=1, and empty columns at end)
    private static final int[] SENSOR_COLUMN_INDICES = {
        2,   // CO(GT)
        3,   // PT08.S1(CO)
        4,   // NMHC(GT)
        5,   // C6H6(GT)
        6,   // PT08.S2(NMHC)
        7,   // NOx(GT)
        8,   // PT08.S3(NOx)
        9,   // NO2(GT)
        10,  // PT08.S4(NO2)
        11,  // PT08.S5(O3)
        12,  // T
        13,  // RH
        14   // AH
    };
    
    public static void main(String[] args) {
        int skipLines = 3000; // Number of lines to skip
        final String bootstrapServers = System.getenv().getOrDefault("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092");
        
        // Create Kafka topics first
        createKafkaTopics(bootstrapServers);
        
        Properties props = new Properties() {{
            put(ProducerConfig.BOOTSTRAP_SERVERS_CONFIG, bootstrapServers);
            put(ProducerConfig.KEY_SERIALIZER_CLASS_CONFIG, StringSerializer.class);
            put(ProducerConfig.VALUE_SERIALIZER_CLASS_CONFIG, StringSerializer.class);
            put(ProducerConfig.ACKS_CONFIG, "1");
        }};

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
                // Parse CSV line and send each sensor value to its corresponding topic
                String[] values = line.split(";");
                if (values.length >= 15) { // Ensure we have enough columns
                    for (int i = 0; i < SENSOR_TOPICS.length; i++) {
                        String sensorValue = values[SENSOR_COLUMN_INDICES[i]].trim();
                            String key = (values[0] + " " + values[1]);
                            String message = sensorValue;
                            
                            ProducerRecord<String, String> sensorRecord = new ProducerRecord<>(SENSOR_TOPICS[i], key, message);
                            RecordMetadata metadata = producer.send(sensorRecord).get();
                            System.out.printf("Sent %s=%s to topic %s partition %d offset %d\n", 
                                SENSOR_TOPICS[i], sensorValue, metadata.topic(), metadata.partition(), metadata.offset());

                    }
                    producer.flush();
                    Thread.sleep(1000); // 1 second delay between rows
                }
            }
            reader.close();
            producer.close();
        } catch (Exception e) {
            System.err.println("Failed to stream CSV to Kafka: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    private static void createKafkaTopics(String bootstrapServers) {
        // Admin client properties
        Properties adminProps = new Properties();
        adminProps.put(AdminClientConfig.BOOTSTRAP_SERVERS_CONFIG, bootstrapServers);
        
        try (AdminClient adminClient = AdminClient.create(adminProps)) {
            List<NewTopic> newTopics = new ArrayList<>();
            
            // Create NewTopic objects with configuration
            for (String topicName : SENSOR_TOPICS) {
                NewTopic newTopic = new NewTopic(topicName, 1, (short) 1);
                newTopics.add(newTopic);
            }
            
            try {
                // Create topics
                adminClient.createTopics(newTopics).all().get();
                System.out.println("Successfully created " + newTopics.size() + " sensor topics:");
                for (String topicName : SENSOR_TOPICS) {
                    System.out.println("  - " + topicName);
                }
            } catch (InterruptedException | ExecutionException e) {
                if (e.getCause() instanceof TopicExistsException) {
                    System.out.println("Some topics already exist. Checking existing topics...");
                    checkExistingTopics(adminClient);
                } else {
                    System.err.println("Error creating topics: " + e.getMessage());
                    e.printStackTrace();
                }
            }
        } catch (Exception e) {
            System.err.println("Failed to create topics: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    private static void checkExistingTopics(AdminClient adminClient) {
        try {
            Set<String> existingTopics = adminClient.listTopics().names().get();
            System.out.println("Existing sensor topics:");
            for (String topicName : SENSOR_TOPICS) {
                if (existingTopics.contains(topicName)) {
                    System.out.println("  ✓ " + topicName + " (exists)");
                } else {
                    System.out.println("  ✗ " + topicName + " (missing)");
                }
            }
        } catch (InterruptedException | ExecutionException e) {
            System.err.println("Error checking existing topics: " + e.getMessage());
            e.printStackTrace();
        }
    }
}