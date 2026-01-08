#!/bin/bash
set -euo pipefail

# =============================================================================
# KNNWeight Filter Optimization Benchmark Script
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${REPO_ROOT}/benchmark_filter_${TIMESTAMP}.log"
RESULTS_DIR="${REPO_ROOT}/benchmark_results_${TIMESTAMP}"

# Configuration
DATASET_PATH="/Users/finnrobl/Downloads/efficient-filters-test/cohere-1m-with-filtering.hdf5"
QUERY_PATH="/Users/finnrobl/Downloads/efficient-filters-test/answer/answer-filter01pct.hdf5"
WORKLOAD_PATH="/Users/finnrobl/Documents/opensearch-benchmark-workloads/vectorsearch"
DATA_DIR="${RESULTS_DIR}/opensearch-data"
OSB_RESULTS_DIR="${RESULTS_DIR}/osb-results"
NUM_RUNS=3
OPENSEARCH_PORT=9200

# Branches (pre-instrumented)
BASELINE_BRANCH="main-instrumented"
OPTIMIZED_BRANCH="efficient-filters-instrumented"

GRADLE_PID=""

log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $1"
    echo "$msg" | tee -a "$LOG_FILE"
}

log_section() {
    log "============================================================================="
    log "$1"
    log "============================================================================="
}

cleanup() {
    log "Cleaning up..."
    stop_opensearch || true
}
trap cleanup EXIT

setup_directories() {
    log_section "Setting up directories"
    mkdir -p "$RESULTS_DIR" "$OSB_RESULTS_DIR" "$DATA_DIR"
    log "Results directory: $RESULTS_DIR"
    log "Data directory: $DATA_DIR"
    
    log "Installing Python dependencies..."
    pip install pandas
}

check_prerequisites() {
    log_section "Checking prerequisites"
    
    for path in "$DATASET_PATH" "$QUERY_PATH"; do
        if [[ ! -f "$path" ]]; then
            log "ERROR: File not found: $path"
            exit 1
        fi
        log "Found: $path"
    done
    
    if [[ ! -d "$WORKLOAD_PATH" ]]; then
        log "ERROR: Workload not found: $WORKLOAD_PATH"
        exit 1
    fi
    log "Workload found: $WORKLOAD_PATH"
    
    if ! command -v opensearch-benchmark &> /dev/null; then
        log "ERROR: opensearch-benchmark not found"
        exit 1
    fi
    log "opensearch-benchmark found"
    log "Prerequisites check passed"
}

checkout_branch() {
    local branch="$1"
    log "Checking out branch: $branch"
    cd "$REPO_ROOT"
    git stash --quiet 2>/dev/null || true
    git checkout "$branch"
}

start_opensearch() {
    local label="$1"
    log_section "Starting OpenSearch ($label) via gradlew run"
    
    cd "$REPO_ROOT"
    
    # Start OpenSearch with shared data directory
    ./gradlew run --data-dir "$DATA_DIR" --no-daemon > "${RESULTS_DIR}/opensearch_${label}.log" 2>&1 &
    GRADLE_PID=$!
    log "Started gradlew run with PID: $GRADLE_PID (data-dir: $DATA_DIR)"
    
    # Wait for startup
    log "Waiting for OpenSearch to start..."
    local max_wait=300
    local waited=0
    while ! curl -s "http://localhost:${OPENSEARCH_PORT}" > /dev/null 2>&1; do
        sleep 5
        waited=$((waited + 5))
        if [[ $waited -ge $max_wait ]]; then
            log "ERROR: OpenSearch failed to start within ${max_wait}s"
            log "Last 50 lines of log:"
            tail -50 "${RESULTS_DIR}/opensearch_${label}.log" | tee -a "$LOG_FILE"
            exit 1
        fi
        log "Waiting... (${waited}s)"
    done
    log "OpenSearch started successfully"
    curl -s "http://localhost:${OPENSEARCH_PORT}" | tee -a "$LOG_FILE"
    echo "" >> "$LOG_FILE"
}

stop_opensearch() {
    log "Stopping OpenSearch..."
    if [[ -n "$GRADLE_PID" ]] && kill -0 "$GRADLE_PID" 2>/dev/null; then
        kill "$GRADLE_PID" 2>/dev/null || true
        sleep 5
        kill -9 "$GRADLE_PID" 2>/dev/null || true
    fi
    # Kill any remaining OpenSearch/Java processes from gradlew
    pkill -f "opensearch" 2>/dev/null || true
    pkill -f "org.opensearch.bootstrap.OpenSearch" 2>/dev/null || true
    sleep 3
    GRADLE_PID=""
    log "OpenSearch stopped"
}

run_index() {
    log_section "Indexing data via OSB"
    
    cat > "${OSB_RESULTS_DIR}/params_ingest.json" <<EOF
{
    "target_index_name": "target_index",
    "target_field_name": "target_field",
    "target_index_body": "indices/faiss-index-filtering.json",
    "target_index_primary_shards": 1,
    "target_index_replica_shards": 0,
    "target_index_dimension": 768,
    "target_index_space_type": "innerproduct",
    "target_index_bulk_size": 500,
    "target_index_bulk_index_data_set_format": "hdf5",
    "target_index_bulk_index_data_set_path": "${DATASET_PATH}",
    "target_dataset_filter_attributes": ["filter01pct", "filter1pct", "filter5pct", "filter10pct", "filter25pct", "filter50pct", "filter75pct", "filter90pct", "filter99pct"],
    "target_index_bulk_indexing_clients": 4
}
EOF
    
    opensearch-benchmark execute-test \
        --target-hosts "localhost:${OPENSEARCH_PORT}" \
        --workload-path "${WORKLOAD_PATH}" \
        --workload-params "${OSB_RESULTS_DIR}/params_ingest.json" \
        --pipeline benchmark-only \
        --test-procedure no-train-test-index-only \
        --kill-running-processes \
        2>&1 | tee -a "$LOG_FILE"
}

force_merge() {
    log "Force merging to 5 segments..."
    curl -s -X POST "http://localhost:${OPENSEARCH_PORT}/target_index/_forcemerge?max_num_segments=5&wait_for_completion=true" | tee -a "$LOG_FILE"
    echo "" >> "$LOG_FILE"
    curl -s -X POST "http://localhost:${OPENSEARCH_PORT}/target_index/_refresh" >> "$LOG_FILE"
    log "Segments after merge:"
    curl -s "http://localhost:${OPENSEARCH_PORT}/_cat/segments/target_index?v" | tee -a "$LOG_FILE"
}

run_search() {
    local label="$1"
    local run_num="$2"
    local output_file="${OSB_RESULTS_DIR}/${label}_run${run_num}.csv"
    
    log "Running search: $label run $run_num"
    
    cat > "${OSB_RESULTS_DIR}/params_search.json" <<EOF
{
    "target_index_name": "target_index",
    "target_field_name": "target_field",
    "target_index_dimension": 768,
    "query_k": 100,
    "query_body": { "docvalue_fields": ["_id"], "stored_fields": "_none_" },
    "filter_type": "efficient",
    "filter_body": { "bool": { "filter": [{ "term": { "filter01pct": "true" } }] } },
    "query_data_set_format": "hdf5",
    "query_data_set_path": "${QUERY_PATH}",
    "search_clients": 8
}
EOF
    
    opensearch-benchmark execute-test \
        --target-hosts "localhost:${OPENSEARCH_PORT}" \
        --workload-path "${WORKLOAD_PATH}" \
        --workload-params "${OSB_RESULTS_DIR}/params_search.json" \
        --pipeline benchmark-only \
        --test-procedure search-only \
        --kill-running-processes \
        --results-format csv \
        --results-file "$output_file" \
        2>&1 | tee -a "$LOG_FILE"
    
    log "Results saved: $output_file"
}

main() {
    log_section "KNNWeight Filter Optimization Benchmark"
    log "Timestamp: $TIMESTAMP"
    log "Baseline: $BASELINE_BRANCH"
    log "Optimized: $OPTIMIZED_BRANCH"
    
    setup_directories
    check_prerequisites
    
    # === INDEX ONCE with baseline ===
    checkout_branch "$BASELINE_BRANCH"
    start_opensearch "baseline"
    run_index
    force_merge
    stop_opensearch
    
    # === BASELINE SEARCH RUNS (reuse indexed data) ===
    log_section "Running baseline search benchmarks"
    start_opensearch "baseline"
    for run in $(seq 1 $NUM_RUNS); do
        run_search "baseline" "$run"
        sleep 5
    done
    stop_opensearch
    
    # === OPTIMIZED SEARCH RUNS (reuse same indexed data) ===
    log_section "Running optimized search benchmarks"
    checkout_branch "$OPTIMIZED_BRANCH"
    start_opensearch "optimized"
    for run in $(seq 1 $NUM_RUNS); do
        run_search "optimized" "$run"
        sleep 5
    done
    stop_opensearch
    
    log_section "Benchmark Complete"
    log "Results: $OSB_RESULTS_DIR"
    log "Log: $LOG_FILE"
}

main "$@"
