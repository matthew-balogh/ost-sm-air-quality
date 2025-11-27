
## Docker compose

```bash
# first start
./start-streaming.sh

# delete the compose stack to reset and start again
docker compose down
./start-streaming.sh
```

![alt text](_images/image.png)

## Kafka

1. Verify messages are getting produced at [localhost:8080/ui/clusters/local/all-topics/co_gt/messages](http://localhost:8080/ui/clusters/local/all-topics/co_gt/messages)

![alt text](_images/image-2.png)

## Influxdb

1. Visit influxdb ui at [http://localhost:8888/configure/servers](http://localhost:8888/configure/servers)
2. Configure a server
3. Check records at [http://localhost:8888/querytool/builder](http://localhost:8888/querytool/builder)

![alt text](_images/image-1.png)

![alt text](_images/image-6.png)

## Grafana

1. Add influxdb data source
2. Add query to dashboard panel

![alt text](_images/image-3.png)

![alt text](_images/image-4.png)

![alt text](_images/image-5.png)