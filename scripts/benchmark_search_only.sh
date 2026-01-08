#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="${REPO_ROOT}/benchmark_results_${TIMESTAMP}"
OSB_RESULTS_DIR="${RESULTS_DIR}/osb-results"

# Reuse existing data dir
DATA_DIR="/Users/finnrobl/Documents/k-NN-2/k-NN/benchmark_results_20260106_155016/opensearch-data"
QUERY_PATH="/Users/finnrobl/Downloads/efficient-filters-test/answer/answer-filter01pct.hdf5"
WORKLOAD_PATH="/Users/finnrobl/Documents/opensearch-benchmark-workloads/vectorsearch"
NUM_RUNS=3
OPENSEARCH_PORT=9200

# Order: optimized first, then baseline
FIRST_BRANCH="efficient-filters-instrumented"
FIRST_LABEL="optimized"
SECOND_BRANCH="main-instrumented"
SECOND_LABEL="baseline"

GRADLE_PID=""

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"; }

stop_opensearch() {
    if [[ -n "$GRADLE_PID" ]] && kill -0 "$GRADLE_PID" 2>/dev/null; then
        kill "$GRADLE_PID" 2>/dev/null || true
        sleep 5
    fi
    pkill -f "opensearch" 2>/dev/null || true
    sleep 3
}
trap stop_opensearch EXIT

start_opensearch() {
    local branch=$1
    log "Checking out $branch"
    cd "$REPO_ROOT" && git checkout "$branch"
    
    log "Building and starting OpenSearch with existing data"
    ./gradlew run --data-dir "$DATA_DIR" &
    GRADLE_PID=$!
    
    for i in {1..120}; do
        if curl -s "http://localhost:${OPENSEARCH_PORT}" > /dev/null 2>&1; then
            log "OpenSearch started"
            return 0
        fi
        sleep 5
    done
    log "ERROR: OpenSearch failed to start"
    exit 1
}

run_search() {
    local label=$1
    local run_num=$2
    log "Running search: $label run $run_num"
    
    opensearch-benchmark execute-test \
        --workload-path="$WORKLOAD_PATH" \
        --workload-params='{
            "target_index_name": "target_index",
            "target_field_name": "target_field",
            "target_index_dimension": 768,
            "target_index_space_type": "innerproduct",
            "query_k": 100,
            "query_body": { "docvalue_fields": ["_id"], "stored_fields": "_none_" },
            "filter_type": "efficient",
            "filter_body": { "bool": { "filter": [{ "term": { "filter01pct": "true" } }] } },
            "query_data_set_format": "hdf5",
            "query_data_set_path": "'"${QUERY_PATH}"'",
            "search_clients": 8
        }' \
        --target-hosts="localhost:${OPENSEARCH_PORT}" \
        --pipeline=benchmark-only \
        --test-procedure=search-only \
        --results-format=csv \
        --results-file="${OSB_RESULTS_DIR}/${label}_run${run_num}.csv"
}

main() {
    mkdir -p "$OSB_RESULTS_DIR"
    log "Results: $RESULTS_DIR"
    log "Reusing data: $DATA_DIR"

    # First version
    start_opensearch "$FIRST_BRANCH"
    for i in $(seq 1 $NUM_RUNS); do run_search "$FIRST_LABEL" "$i"; done
    stop_opensearch

    # Second version  
    start_opensearch "$SECOND_BRANCH"
    for i in $(seq 1 $NUM_RUNS); do run_search "$SECOND_LABEL" "$i"; done
    stop_opensearch

    log "Complete. Results: $OSB_RESULTS_DIR"
}

main
