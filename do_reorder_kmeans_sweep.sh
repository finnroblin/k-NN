#!/bin/bash
set -euo pipefail

# --- Configuration ---
DISTRO="/home/ec2-user/k-NN-finn/build/testclusters/integTest-0/distro/3.6.0-ARCHIVE"
LIB_PATH="${DISTRO}/plugins/opensearch-knn"
CLASSPATH="${DISTRO}/lib/*:${DISTRO}/plugins/opensearch-knn/*"
DATA_DIR="${DISTRO}/data"
BASELINE_DIR="/home/ec2-user/before-reordering/data/nodes"
REORDERED_BASE="/home/ec2-user/reordering-sweep-data"
CENTROIDS=(500 1000 1500 2000 2500 3000)
RESULTS_DIR="benchmark_results/kmeans_$(date +%Y%m%d_%H%M%S)"

mkdir -p "$RESULTS_DIR"

for k in "${CENTROIDS[@]}"; do
    echo "===== Running k=$k ====="
    LOG_FILE="${RESULTS_DIR}/kmeans_k${k}.log"

    # Restore from baseline before each run
    echo "Restoring shard data from baseline..."
    rm -rf "$DATA_DIR"
    cp -r "$BASELINE_DIR" "$DATA_DIR"

    java -Djava.library.path="$LIB_PATH" \
         -Dknn.kmeans.numClusters="$k" \
         -cp "$CLASSPATH" \
         org.opensearch.knn.memoryoptsearch.faiss.reorder.ReorderAllWithKMeans \
         2>&1 | tee "$LOG_FILE"

    # Copy reordered data to a separate directory per centroid count
    REORDERED_DIR="${REORDERED_BASE}/kmeans_k${k}/data"
    echo "Saving reordered data to ${REORDERED_DIR} ..."
    rm -rf "${REORDERED_BASE}/kmeans_k${k}"
    mkdir -p "${REORDERED_BASE}/kmeans_k${k}"
    cp -r "$DATA_DIR" "$REORDERED_DIR"

    echo "--- k=$k complete, log: $LOG_FILE ---"
    echo
done

echo "All runs complete. Results in: $RESULTS_DIR"
echo ""
echo "Reordered data saved under: ${REORDERED_BASE}/"
echo "To run search against a specific k:"
echo "  rm -rf /home/ec2-user/before-reordering/after-reordering/data"
echo "  cp -r ${REORDERED_BASE}/kmeans_k<N>/data /home/ec2-user/before-reordering/after-reordering/data"
