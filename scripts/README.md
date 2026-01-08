# Benchmark Scripts

## Overview

These scripts benchmark the KNNWeight filter optimization that avoids BitSet creation for exact search queries.

## Running Locally

### Prerequisites
- OpenSearch Benchmark (`pip install opensearch-benchmark`)
- Dataset files in `/Users/finnrobl/Downloads/efficient-filters-test/`
- Workload repo at `/Users/finnrobl/Documents/opensearch-benchmark-workloads/vectorsearch`

### Full Benchmark (Index + Search)
```bash
./scripts/benchmark_filter_optimization.sh
```
This will:
1. Index 1M vectors once (baseline branch)
2. Force merge to 5 segments
3. Run 3 search benchmarks on baseline
4. Switch to optimized branch, reuse indexed data
5. Run 3 search benchmarks on optimized

### Search-Only Benchmark (Reuse Existing Data)
```bash
./scripts/benchmark_search_only.sh
```
Edit the script to set `DATA_DIR` to an existing indexed data directory from a previous run.

### Analyze Results
```bash
python scripts/analyze_benchmark.py
```

## Scripts

| Script | Purpose |
|--------|---------|
| `benchmark_filter_optimization.sh` | Full benchmark: index once, search both branches |
| `benchmark_search_only.sh` | Search-only using pre-indexed data |
| `benchmark_filter10pct.sh` | Archived - used wrong filter selectivity |
| `analyze_benchmark.py` | Parse OSB CSV results, compute averages |

## Docker Status

Docker configs were created in `opensearch-knn-single-node-experiments/demo/local/` but are **not equivalent** to the local scripts.

### What Docker Does
- Runs full index + search workload each time
- Single branch per run

### What Local Scripts Do
- Index once, reuse data across branch switches
- Search-only benchmarks isolate the optimization impact
- Multiple runs per branch for statistical validity

### To Match Local Workflow in Docker
Would require:
1. Two pre-built images (baseline + optimized branches)
2. Persistent volume with indexed data
3. OSB `search-only` test procedure
4. Wrapper script to swap images while preserving data volume

This is not yet implemented.
