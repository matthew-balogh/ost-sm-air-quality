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
import java.io.FileReader;
import java.io.File;
import java.nio.file.Paths;

public class Main {
    static String topic = "sensor-data";
    
    // List of topics based on CSV columns (excluding Date, Time, empty columns, and columns with "_gt" in name)
    public static final String[] SENSOR_TOPICS = {
        "pt08_s1_co",      // PT08.S1(CO)
        "pt08_s2_nmhc",    // PT08.S2(NMHC)
        "pt08_s3_nox",     // PT08.S3(NOx)
        "pt08_s4_no2",     // PT08.S4(NO2)
        "pt08_s5_o3",      // PT08.S5(O3)
        "t",               // T
        "rh",              // RH
        "ah"               // AH
    };
    
    // CSV column indices (0-based, excluding Date=0, Time=1, empty columns at end, and columns with "_gt" in name)
    private static final int[] SENSOR_COLUMN_INDICES = {
        3,   // PT08.S1(CO)
        6,   // PT08.S2(NMHC)
        8,   // PT08.S3(NOx)
        10,  // PT08.S4(NO2)
        11,  // PT08.S5(O3)
        12,  // T
        13,  // RH
        14   // AH
    };
    
    private static final int[] ANOMALY_SENSOR_INDICES = {0, 1, 2, 3, 4}; // pt08_s1_co, pt08_s2_nmhc, pt08_s3_nox, pt08_s4_no2, pt08_s5_o3
    
    public static void main(String[] args) {
        int skipLines = 780; // Number of lines to skip
        final String bootstrapServers = System.getenv().getOrDefault("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092");
        
        List<String[]> anomaliesData = loadAnomaliesData();
        if (anomaliesData.isEmpty()) {
            System.err.println("Warning: Could not load anomalies.csv. Continuing without anomaly injection.");
        }
        
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
            int anomalyIndex = 0;
            // Skip the first 'skipLines' lines
            for (int i = 0; i < skipLines; i++) {
                reader.readLine();
            }
            while ((line = reader.readLine()) != null) {
                rowNum++;
                // Parse CSV line and send each sensor value to its corresponding topic
                String[] values = line.split(";");
                if (values.length >= 15) { // Ensure we have enough columns
                    boolean useAnomaly = (rowNum % 10 == 0) && !anomaliesData.isEmpty();
                    
                    if (useAnomaly) {
                        String[] anomalyRow = anomaliesData.get(anomalyIndex % anomaliesData.size());
                        anomalyIndex++;
                        
                        for (int i = 0; i < SENSOR_TOPICS.length; i++) {
                            String sensorValue;
                            if (i < ANOMALY_SENSOR_INDICES.length && i < anomalyRow.length) {
                                sensorValue = anomalyRow[i].trim();
                                if (sensorValue.isEmpty()) {
                                    sensorValue = values[SENSOR_COLUMN_INDICES[i]].trim();
                                }
                            } else {
                                sensorValue = values[SENSOR_COLUMN_INDICES[i]].trim();
                            }
                            
                            String key = (values[0] + " " + values[1]);
                            String message = String.valueOf(Double.parseDouble(sensorValue.replace(',', '.')));
                            ProducerRecord<String, String> sensorRecord = new ProducerRecord<>(SENSOR_TOPICS[i], key, message);
                            RecordMetadata metadata = producer.send(sensorRecord).get();
                            System.out.printf("Sent %s=%s to topic %s partition %d offset %d%s\n", 
                                SENSOR_TOPICS[i], sensorValue, metadata.topic(), metadata.partition(), metadata.offset(),
                                (i < ANOMALY_SENSOR_INDICES.length && useAnomaly) ? " [ANOMALY]" : "");
                        }
                    } else {
                        for (int i = 0; i < SENSOR_TOPICS.length; i++) {
                            String sensorValue = values[SENSOR_COLUMN_INDICES[i]].trim();
                            String key = (values[0] + " " + values[1]);
                            String message = String.valueOf(Double.parseDouble(sensorValue.replace(',', '.')));
                            ProducerRecord<String, String> sensorRecord = new ProducerRecord<>(SENSOR_TOPICS[i], key, message);
                            RecordMetadata metadata = producer.send(sensorRecord).get();
                            System.out.printf("Sent %s=%s to topic %s partition %d offset %d\n", 
                                SENSOR_TOPICS[i], sensorValue, metadata.topic(), metadata.partition(), metadata.offset());
                        }
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

    private static List<String[]> loadAnomaliesData() {
        List<String[]> anomalies = new ArrayList<>();
        try {
            // Try to find Data/anomalies.csv relative to current working directory
            String currentDir = System.getProperty("user.dir");
            String anomaliesPath = Paths.get(currentDir, "Data", "anomalies.csv").toString();
            File anomaliesFile = new File(anomaliesPath);
            
            // If not found, try relative path from project root
            if (!anomaliesFile.exists()) {
                anomaliesFile = new File("Data/anomalies.csv");
            }
            
            if (!anomaliesFile.exists() || !anomaliesFile.isFile()) {
                System.err.println("Could not find Data/anomalies.csv at: " + anomaliesPath);
                System.err.println("Also tried: Data/anomalies.csv");
                return anomalies;
            }
            
            System.out.println("Loading anomalies from: " + anomaliesFile.getAbsolutePath());
            
            BufferedReader reader = new BufferedReader(new FileReader(anomaliesFile));
            String line;
            boolean firstLine = true;
            
            while ((line = reader.readLine()) != null) {
                if (firstLine) {
                    firstLine = false;
                    continue;
                }
                
                if (line.trim().isEmpty()) {
                    continue;
                }
                
                String[] values = line.split(";");
                if (values.length >= 5) {
                    anomalies.add(values);
                }
            }
            reader.close();
            
            System.out.println("Loaded " + anomalies.size() + " anomaly records.");
        } catch (Exception e) {
            System.err.println("Error loading anomalies.csv: " + e.getMessage());
            e.printStackTrace();
        }
        return anomalies;
    }
}