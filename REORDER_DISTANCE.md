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
