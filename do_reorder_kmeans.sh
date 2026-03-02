#!/bin/bash

cp /home/ec2-user/k-NN-finn/jni/build/release/*.so \
   /home/ec2-user/k-NN-finn/build/testclusters/integTest-0/distro/3.6.0-SNAPSHOT/plugins/opensearch-knn/

java -Djava.library.path="/home/ec2-user/k-NN-finn/build/testclusters/integTest-0/distro/3.6.0-SNAPSHOT/plugins/opensearch-knn" \
    -cp "/home/ec2-user/k-NN-finn/build/testclusters/integTest-0/distro/3.6.0-SNAPSHOT/lib/*:/home/ec2-user/k-NN-finn/build/testclusters/integTest-0/distro/3.6.0-SNAPSHOT/plugins/opensearch-knn/*" \
    org.opensearch.knn.memoryoptsearch.faiss.reorder.ReorderAllWithKMeans