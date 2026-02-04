#!/bin/bash

cp /home/ec2-user/k-NN/jni/build/release/*.so \
   /home/ec2-user/k-NN/build/testclusters/integTest-0/distro/3.5.0-ARCHIVE/plugins/opensearch-knn/

java -cp \
        "/home/ec2-user/k-NN/build/testclusters/integTest-0/distro/3.5.0-ARCHIVE/lib/*:/home/ec2-user/k-NN/build/testclusters/integTest-0/distro/3.5.0-ARCHIVE/plugins/opensearch-knn/*" \
          org.opensearch.knn.memoryoptsearch.faiss.reorder.ReorderAllWithKMeans
