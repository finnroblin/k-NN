# K-Means Reorder Benchmarking Plan

## Objective

Measure the impact of varying the number of K-Means centroids on reorder quality and performance. Sweep `k` across: **500, 1000, 1500, 2000, 2500, 3000**.

## Required Code Change

`k` is now directly configurable via `-Dknn.kmeans.numClusters=N` (defaults to 500), no clamping.

## Step 1: Build

```bash
cd /home/ec2-user/k-NN-finn
git pull origin finn-reorder-kmeans
./gradlew :buildJniLib && ./gradlew assemble
mv build/testclusters/integTest-0/distro/3.6.0-SNAPSHOT \
   build/testclusters/integTest-0/distro/3.6.0-ARCHIVE
cp jni/build/release/*.so \
   build/testclusters/integTest-0/distro/3.6.0-ARCHIVE/plugins/opensearch-knn/
```

## Step 2: Run Reorder Sweep

`do_reorder_kmeans_sweep.sh` — runs reorder for each centroid count, saves reordered data per k.

```bash
#!/bin/bash
set -euo pipefail

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

    # Save reordered data for later search benchmarking
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
```

## Step 3: Start OpenSearch Cluster (per k value)

For each centroid count you want to search against:

```bash
# 3a. Replace after-reordering data with the reordered data for this k
rm -rf /home/ec2-user/before-reordering/after-reordering/data/nodes
cp -r /home/ec2-user/reordering-sweep-data/kmeans_k<N>/data \
      /home/ec2-user/before-reordering/after-reordering/data/nodes

# 3b. Signal file (required by cluster startup)
touch /tmp/dododo

# 3c. Prep OpenSearch
cd /home/ec2-user/k-NN-finn/build/testclusters/integTest-0/distro/3.6.0-ARCHIVE
rm -rf data
ln -s /home/ec2-user/before-reordering/data
cp ~/jvm.options ./config/jvm.options
cp ~/opensearch.yml ./config/opensearch.yml

# 3d. Start OpenSearch
./bin/opensearch
```

## Reorder Metrics

Each reorder run logs these timings — extract from the log files:

| Metric | Source (log line) |
|---|---|
| Vec loading time | `vec loading took : Xms` |
| Permutation compute time | `permutation took : Xms` |
| Faiss ID update time | `faiss id update took : Xms` |
| Total reorder time | `K-Means Reordering took : Xms` |
| Vec transform time | `Transforming .vec took Xms` |

## Parsing Reorder Results

```bash
#!/bin/bash
# parse_kmeans_results.sh <results_dir>
DIR="${1:?Usage: parse_kmeans_results.sh <results_dir>}"

printf "%-6s  %12s  %12s  %12s  %12s  %12s\n" \
       "k" "vec_load" "permutation" "faiss_id" "total_reorder" "vec_transform"

for f in "$DIR"/kmeans_k*.log; do
    k=$(echo "$f" | grep -oP 'k\K[0-9]+')
    vec_load=$(grep -oP 'vec loading took : \K[0-9.]+' "$f" | tail -1)
    perm=$(grep -oP 'permutation took : \K[0-9.]+' "$f" | tail -1)
    faiss=$(grep -oP 'faiss id update took : \K[0-9.]+' "$f" | tail -1)
    total=$(grep -oP 'K-Means Reordering took : \K[0-9.]+' "$f" | tail -1)
    vec_tx=$(grep -oP 'Transforming .vec took \K[0-9.]+' "$f" | tail -1)
    printf "%-6s  %12s  %12s  %12s  %12s  %12s\n" \
           "$k" "${vec_load}ms" "${perm}ms" "${faiss}ms" "${total}ms" "${vec_tx}ms"
done
```

## Notes

- `k` is directly configurable via `-Dknn.kmeans.numClusters=N` with no clamping.
- Iterations fixed at `DEFAULT_NUM_ITERATIONS = 25`. Can be parameterized similarly if needed.
- `switchFiles` overwrites index files in place — the sweep restores from baseline before each run.
- Reordered data is saved to `/home/ec2-user/reordering-sweep-data/kmeans_k<N>/data` for each centroid count.
- Page faults should be measured during the search workload, not during reordering.
