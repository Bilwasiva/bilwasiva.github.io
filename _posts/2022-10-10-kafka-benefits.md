---
layout: post
title: Unlocking the Power of Real-time Data Streaming with Apache Kafka
image: /img/apache-kafka.png
published: true
---

80% of the Fortune 100 companies use Apache Kafka for some use case or other. Let's see how it works

1. Apache Kafka is a Distributed, Replicated Messaging Queue. It functions like a commit log.

2. It is completely open source and you can download it directly. But to use it, you need to create an Apache Kafka cluster. Because running Apache Kafka on just one system won’t work in a distributed environment.

3. A cluster of Apache Kafka contains multiple servers. Each server is called a broker and stores the data. But where is the data stored? Apache Kafka makes use of secondary storage for storing the data. A lot of people have apprehensions about hard disks being slower than the main memory. You can make this access faster by writing and reading from sequential memory locations rather than random locations. This is how Kafka stores data. How is this access sequential ? We will see below.

4. Regarding storage in Kafka, you’ll always hear two terms - Partition and Topic. Partitions are the units of storage in Kafka for messages. And Topic can be thought of as being a container in which these partitions lie.

5. Whenever you create a topic in Kafka, it creates the directories equal to the number of partitions you have specified - One directory for one partition of the topic. In Kafka, the topic is more of a logical grouping than anything else, and that the Partition is the actual unit of storage in Kafka. That is what is physically stored on the disk. 

6. Each partition is further subdivided into segments. Each segment is a log file containing the incoming messages. Each message which is stored in the log file contains the actual message along with the offset(number of messages in the file + 1) at which it occurs.

7. The messages as they come are written sequentially in one of the partitions for that topic.Each partition can be consumed by only one consumer ay a time.(this is a kafka requirement). 

8. A common operation in Kafka is to read the message at a particular offset. How will you find this offset? Scanning the log file ? But, it will take a lot of time. This is where the index file comes to help which stores the physical address for each offset.

9. Kafka does not always access disk sequentially but it does some things that make it much more likely that disk access is often sequential. All Kafka messages are stored in larger segment files and since Kafka messages are not deleted when consumed (like in other message brokers) Kafka will not end up creating a fragmented filesystem over time by continuously creating and deleting many variable length files.

10. Instead it creates segment files and then appends them to that file until it reaches 1GB(configurable). When all messages in the segment expire, it deletes the entire segment.

11. Kafka is speedy and fault-tolerant distributed streaming platform. However, there are some situations when messages can disappear. It can happen due to misconfiguration or misunderstanding Kafka's internals.
