# K-Means Reorder Benchmarking Plan

## Objective

Measure the impact of varying the number of K-Means centroids on reorder quality and performance. Sweep `k` across: **500, 1000, 1500, 2000, 2500, 3000**.

## Current State

- `ReorderAllWithKMeans` hardcodes `DEFAULT_NUM_CLUSTERS = 500`.
- The shell script `do_reorder_kmeans.sh` invokes the class with no way to override `k`.

## Required Code Change

Add a system property override for `k` in `ReorderAllWithKMeans.getOrderMap()`:

```java
// Replace:
int k = Math.min(DEFAULT_NUM_CLUSTERS, numVectors / 100);

// With:
int configuredK = Integer.getInteger("knn.kmeans.numClusters", DEFAULT_NUM_CLUSTERS);
int k = Math.min(configuredK, numVectors / 100);
```

## Benchmark Script

`do_reorder_kmeans_sweep.sh` — runs reorder for each centroid count, collecting timing output per run.

```bash
#!/bin/bash
set -euo pipefail

# --- Configuration ---
KNN_HOME="/home/ec2-user/k-NN-gorder"
DISTRO="${KNN_HOME}/build/testclusters/integTest-0/distro/3.5.0-ARCHIVE"
LIB_PATH="${DISTRO}/plugins/opensearch-knn"
CLASSPATH="${DISTRO}/lib/*:${DISTRO}/plugins/opensearch-knn/*"
DATA_DIR="${DISTRO}/data"
BASELINE_DIR="${DATA_DIR}_baseline"
CENTROIDS=(500 1000 1500 2000 2500 3000)
RESULTS_DIR="benchmark_results/kmeans_$(date +%Y%m%d_%H%M%S)"

# Create baseline copy if it doesn't exist
if [ ! -d "$BASELINE_DIR" ]; then
    echo "Creating baseline copy of shard data..."
    cp -r "$DATA_DIR" "$BASELINE_DIR"
fi

mkdir -p "$RESULTS_DIR"

for k in "${CENTROIDS[@]}"; do
    echo "===== Running k=$k ====="
    LOG_FILE="${RESULTS_DIR}/kmeans_k${k}.log"

    # Restore from baseline before each run
    echo "Restoring shard data from baseline..."
    rm -rf "$DATA_DIR"
    cp -r "$BASELINE_DIR" "$DATA_DIR"

    pgfault_start=$(awk '$1=="pgmajfault"{print $2}' /proc/vmstat)

    java -Djava.library.path="$LIB_PATH" \
         -Dknn.kmeans.numClusters="$k" \
         -cp "$CLASSPATH" \
         org.opensearch.knn.memoryoptsearch.faiss.reorder.ReorderAllWithKMeans \
         2>&1 | tee "$LOG_FILE"

    pgfault_end=$(awk '$1=="pgmajfault"{print $2}' /proc/vmstat)
    pgfaults=$((pgfault_end - pgfault_start))
    echo "Major page faults: $pgfaults" | tee -a "$LOG_FILE"

    echo "--- k=$k complete, log: $LOG_FILE ---"
    echo
done

echo "All runs complete. Results in: $RESULTS_DIR"
```

## Metrics to Collect

Each run already prints these timings — extract from the log files:

| Metric | Source (log line) |
|---|---|
| Vec loading time | `vec loading took : Xms` |
| Permutation compute time | `permutation took : Xms` |
| Faiss ID update time | `faiss id update took : Xms` |
| Total reorder time | `K-Means Reordering took : Xms` |
| Vec transform time | `Transforming .vec took Xms` |
| Major page faults | `Major page faults: N` |

Additionally, the permutation file `permutation_kmeans.txt` is saved per shard and can be used for displacement analysis.

## Parsing Results

Quick extraction script to build a summary table:

```bash
#!/bin/bash
# parse_kmeans_results.sh <results_dir>
DIR="${1:?Usage: parse_kmeans_results.sh <results_dir>}"

printf "%-6s  %12s  %12s  %12s  %12s  %12s  %12s\n" \
       "k" "vec_load" "permutation" "faiss_id" "total_reorder" "vec_transform" "pgfaults"

for f in "$DIR"/kmeans_k*.log; do
    k=$(echo "$f" | grep -oP 'k\K[0-9]+')
    vec_load=$(grep -oP 'vec loading took : \K[0-9.]+' "$f" | tail -1)
    perm=$(grep -oP 'permutation took : \K[0-9.]+' "$f" | tail -1)
    faiss=$(grep -oP 'faiss id update took : \K[0-9.]+' "$f" | tail -1)
    total=$(grep -oP 'K-Means Reordering took : \K[0-9.]+' "$f" | tail -1)
    vec_tx=$(grep -oP 'Transforming .vec took \K[0-9.]+' "$f" | tail -1)
    pgfaults=$(grep -oP 'Major page faults: \K[0-9]+' "$f" | tail -1)
    printf "%-6s  %12s  %12s  %12s  %12s  %12s  %12s\n" \
           "$k" "${vec_load}ms" "${perm}ms" "${faiss}ms" "${total}ms" "${vec_tx}ms" "$pgfaults"
done
```

## Execution Steps

1. Apply the system property code change to `ReorderAllWithKMeans.java`.
2. Rebuild: `./gradlew :buildJniLib && ./gradlew assemble`.
3. Copy the built `.so` files to the test cluster plugin dir (as `do_reorder_kmeans.sh` already does).
4. Ensure the index data is present under the test cluster data directory.
5. **Create a baseline copy of the shard data before any reordering** — reordering calls `switchFiles` which overwrites the original `.faiss`, `.vec`, and `.vemf` files in place. Each sweep iteration must start from the original unreordered data.
   ```bash
   DATA_DIR="/home/ec2-user/k-NN-gorder/build/testclusters/integTest-0/distro/3.5.0-ARCHIVE/data"
   cp -r "$DATA_DIR" "${DATA_DIR}_baseline"
   ```
6. Run `bash do_reorder_kmeans_sweep.sh` — the sweep script should restore from baseline before each run.
7. Parse: `bash parse_kmeans_results.sh benchmark_results/kmeans_<timestamp>`.

## Notes

- `k` is now directly configurable via `-Dknn.kmeans.numClusters=N` with no clamping — the value you pass is the value used.
- Iterations are fixed at `DEFAULT_NUM_ITERATIONS = 25`. Can be parameterized similarly if needed.
- **`switchFiles` overwrites index files in place.** You must restore from the baseline copy before each run, otherwise subsequent runs reorder already-reordered data.
