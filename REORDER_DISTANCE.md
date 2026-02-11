# Reorder Distance Analysis

This document tracks the analysis of vector reordering impact on internal ordinal distances.

## Overview

When vectors are reordered (e.g., via G-Order or other graph-based reordering), their internal ordinals change. This utility measures how far vectors move from their original positions.

## Key Concepts

### Permutation Mapping

- `newOrd2Old[i] = j` means the vector at new ordinal `i` was originally at ordinal `j`
- `oldOrd2New[j] = i` means the vector originally at ordinal `j` is now at ordinal `i`

### Displacement

For each vector: `displacement = |newOrd - oldOrd|`

### Baseline (No Reordering)

Without reordering, `newOrd == oldOrd` for all vectors, so:
- Mean displacement: 0
- StdDev displacement: 0

## Usage

### Saving Permutation During Reorder

```java
ReorderOrdMap reorderOrdMap = getOrderMap(...);
ReorderDistanceAnalyzer.savePermutation(
    reorderOrdMap.newOrd2Old, 
    Path.of("/path/to/permutation.txt")
);
```

### Analyzing Permutation

```bash
# Command line
java -cp <classpath> org.opensearch.knn.memoryoptsearch.faiss.reorder.ReorderDistanceAnalyzer permutation.txt
```

```java
// Programmatic
int[] permutation = ReorderDistanceAnalyzer.loadPermutation(Path.of("permutation.txt"));
Stats reordered = ReorderDistanceAnalyzer.computeDisplacementStats(permutation);
Stats baseline = ReorderDistanceAnalyzer.computeBaselineStats(permutation.length);
System.out.println(ReorderDistanceAnalyzer.compareStats(reordered, baseline));
```

## Logging Doc IDs for Rescoring

In `KNNWeight.java`, doc IDs passed to `ExactSearcher` for rescoring are logged at DEBUG level:

```
[KNN] ExactSearcher rescoring docIds count: <count>, field: <field>
```

Enable with log level DEBUG for `org.opensearch.knn.index.query.KNNWeight`.

## Example Output

```
=== Reorder Distance Analysis ===
Baseline (1:1):  count=100000, mean=0.00, stdDev=0.00, min=0, max=0
Reordered:       count=100000, mean=15234.50, stdDev=12456.78, min=0, max=89234
Mean increase:   15234.50
StdDev increase: 12456.78
```

## Interpretation

- **Higher mean displacement**: Vectors moved farther on average from original positions
- **Higher stdDev**: More variance in how far vectors moved
- **Max displacement**: Worst-case movement distance

## Files

| File | Description |
|------|-------------|
| `ReorderAll.java` | Main reordering logic |
| `ReorderOrdMap.java` | Permutation mapping structure |
| `ReorderDistanceAnalyzer.java` | Displacement analysis utility |
| `KNNWeight.java` | Contains rescoring doc ID logging |


## Analysis Run (2026-02-11)

### Results (10110 queries, pageSize=50)
```
=== Reorder Distance Analysis ===
Baseline (1:1):  count=1000000, mean=0.00, stdDev=0.00, min=0, max=0
Reordered:       count=1000000, mean=330943.98, stdDev=235776.86, min=1, max=999251

=== Paired T-Test (Page Faults, pageSize=50) ===
Number of queries: 10110
Mean page faults baseline (oldOrds):  493.39
Mean page faults reordered (newOrds): 468.65
t-statistic: -86.2613
(negative t = reordering reduced page faults)
```

Page fault metric: count of distinct pages (docId / pageSize) touched per query. Lower = better locality.

### Commands Run
```bash
# Analysis script
/Users/finnrobl/Documents/k-NN-2/sift-binary/run_analysis.sh

# Manual run (optional page size as 3rd arg, default=50)
cd /Users/finnrobl/Documents/k-NN-2/k-NN
java -cp build/classes/java/main:build/resources/main \
    org.opensearch.knn.memoryoptsearch.faiss.reorder.ReorderDistanceAnalyzer \
    /Users/finnrobl/Documents/k-NN-2/sift-binary/nodes/0/indices/80darGsMRXy1pVee2ywcvQ/0/index/permutation.txt \
    /Users/finnrobl/Documents/k-NN-2/sift-binary/query_doc_ids/exactsearcher_docids_shard_0.txt \
    50
```
