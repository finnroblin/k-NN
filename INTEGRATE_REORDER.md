# Feature

There is currently reordering code present in the k-NN plugin. However , it is implemented as a separate task, accessible through the do_reorder_bp.sh script. 

There is detailed information about reordering present at /Users/finnrobl/Documents/k-NN-2/vector-reorder/PORT_REORDER_TO_KNN.md . Read that.

We now need to implement reordering during the segment creation process.

The contract should be:
- When a segment is made searchable, reordering should apply.
- Reordering should apply to every segment with size greater than 10k vectors.
- Reordering should use a variable number of CPU cores , currently just using a default constant. 

Current thought: 
Put it in the vectors flush path, as opposed to the merge path or the refresh path.

Please fill out an implementation plan. Start by going through the flush, merge, and refresh pipelines (may also need to look at the OpenSearch repo under /Users/finnrobl/Documents/OpenSearch).

Put information about each of the entrypoints of this function and when they are triggered in the lifetime of an opensearch cluster (e.g. after indexing but before search, during force merge, etc).

Put all of this information in the document, and then figure out additional tasks for integration. We will do bipartite reordering for now, but make it pluggable so I can do kmeans reordering instead.

Also include test plans.

---

## Pipeline Analysis: Flush, Merge, and Refresh in OpenSearch + k-NN

### 1. Flush Pipeline

**When triggered:** When the in-memory indexing buffer (Lucene's `IndexWriter` buffer) is written to a new on-disk segment. This happens:
- Automatically when the indexing buffer exceeds `indices.memory.index_buffer_size` (default 10% of heap)
- Explicitly via the `_flush` API
- Before a `_refresh` if there are buffered docs
- During shard recovery

**OpenSearch core entrypoint:**
- `IndexShard.flush(FlushRequest)` → `InternalEngine.flush(boolean force, boolean waitIfOngoing)` → Lucene `IndexWriter.flush()` → triggers codec's `KnnVectorsWriter.flush()`

**k-NN entrypoint:**
- `NativeEngines990KnnVectorsWriter.flush(int maxDoc, Sorter.DocMap sortMap)`
  - Iterates over each `NativeEngineFieldVectorsWriter` (one per knn_vector field)
  - For each field: trains quantization if needed, then calls `NativeIndexWriter.flushIndex()`
  - `NativeIndexWriter.flushIndex()` → `buildAndWriteIndex()` → writes `.faiss` / engine file via `NativeIndexBuildStrategy.buildAndWriteIndex()`
  - The `flatVectorsWriter.flush()` call writes the `.vec` flat vectors file (Lucene99FlatVectorsWriter)
  - Stat tracked: `KNNGraphValue.REFRESH_TOTAL_TIME_IN_MILLIS`

**Segment lifecycle context:** After flush, the segment exists on disk but is NOT yet searchable. It becomes searchable after a refresh opens a new `IndexSearcher` over the latest segments.

**Typical segment sizes at flush:** Small-to-medium. Depends on buffer size and indexing rate. Often 1k–100k+ vectors per flush depending on workload.

### 2. Merge Pipeline

**When triggered:** When Lucene's `MergePolicy` (typically `TieredMergePolicy`) decides to combine multiple smaller segments into a larger one. This happens:
- Automatically in the background via `ConcurrentMergeScheduler` after flushes
- Explicitly via `_forcemerge` API (force merge to N segments)
- During shard recovery

**OpenSearch core entrypoint:**
- Background: `InternalEngine.EngineMergeScheduler` (extends `OpenSearchConcurrentMergeScheduler`) runs merges on dedicated threads
- Explicit: `IndexShard.forceMerge(ForceMergeRequest)` → `InternalEngine.forceMerge()` → Lucene `IndexWriter.forceMerge()`
- Lucene calls codec's `KnnVectorsWriter.mergeOneField()` for each knn_vector field in the merged segment

**k-NN entrypoint:**
- `NativeEngines990KnnVectorsWriter.mergeOneField(FieldInfo fieldInfo, MergeState mergeState)`
  - Merges flat vectors via `flatVectorsWriter.mergeOneField()` (writes merged `.vec`)
  - Reads merged vector values from all source segments
  - Trains quantization if needed, then calls `NativeIndexWriter.mergeIndex()`
  - `NativeIndexWriter.mergeIndex()` → `buildAndWriteIndex()` → writes new `.faiss` engine file
  - Stat tracked: `KNNGraphValue.MERGE_TOTAL_TIME_IN_MILLIS`

**Segment lifecycle context:** Merge creates a new, larger segment from multiple existing segments. The old segments are deleted after the merge completes and a refresh makes the new segment visible.

**Typical segment sizes at merge:** Medium-to-large. Tiered merge policy creates segments up to `max_merged_segment` (default 5GB). Force merge can create very large segments (entire index in one segment).

### 3. Refresh Pipeline

**When triggered:** When a new `IndexSearcher` is opened over the latest committed segments, making recently flushed/merged segments searchable. This happens:
- Automatically every `index.refresh_interval` (default 1 second)
- Explicitly via `_refresh` API
- Implicitly before search if `?refresh=true`

**OpenSearch core entrypoint:**
- `IndexShard.refresh(String source)` → `InternalEngine.refresh(String source)` → Lucene `ReferenceManager.maybeRefresh()` → opens new `DirectoryReader`
- Refresh listeners are notified: `RefreshListeners`, `CheckpointRefreshListener`, `RemoteStoreRefreshListener`

**k-NN entrypoint:**
- Refresh does NOT invoke any k-NN codec writer methods. It only opens a new reader over existing segments.
- k-NN's `NativeEngines990KnnVectorsReader` is instantiated when a new `DirectoryReader` opens a segment for the first time. This loads the `.faiss` index into the native memory cache.

**Segment lifecycle context:** Refresh makes segments searchable. No new segment files are written during refresh.

### Summary Table

| Pipeline | Trigger | Writes new segment files? | k-NN codec method | Typical vector count | Searchable after? |
|----------|---------|--------------------------|-------------------|---------------------|-------------------|
| Flush | Buffer full, explicit `_flush` | Yes (new segment) | `NativeEngines990KnnVectorsWriter.flush()` | 1k–100k+ | After next refresh |
| Merge | Background merge policy, `_forcemerge` | Yes (merged segment) | `NativeEngines990KnnVectorsWriter.mergeOneField()` | 10k–millions | After next refresh |
| Refresh | Timer (1s default), explicit `_refresh` | No | None (reader only) | N/A | Immediately |

---

## Recommended Integration Point

**Primary: Flush path** (`NativeEngines990KnnVectorsWriter.flush()`)

Rationale:
- Flush is where new segment files (`.vec`, `.faiss`) are first written to disk
- Reordering the `.vec` and `.faiss` files at flush time means every segment is reordered before it ever becomes searchable
- The flat vectors are already materialized in memory during flush (via `NativeEngineFieldVectorsWriter.getVectors()`), so computing a reorder permutation is cheap relative to the I/O
- Flush happens on the indexing thread, so adding reorder latency here is acceptable (it's already blocking)

**Secondary: Merge path** (`NativeEngines990KnnVectorsWriter.mergeOneField()`)

Rationale:
- Merged segments are also new segments that need reordering
- Without merge-path reordering, a force-merged segment would lose its reorder properties
- Merge already reads all vectors from source segments, so the data is available

**Not recommended: Refresh path**
- Refresh does not write segment files; it only opens readers
- No codec writer methods are invoked during refresh
- Would require a fundamentally different architecture (post-hoc rewriting)

---

## Implementation Plan

### TODO 1: Define `VectorReorderStrategy` interface (pluggable reordering)

Create a strategy interface so BP and KMeans (and future strategies) are interchangeable.

**File:** `k-NN/src/main/java/org/opensearch/knn/memoryoptsearch/faiss/reorder/VectorReorderStrategy.java`

```java
public interface VectorReorderStrategy {
    /**
     * Compute a permutation array mapping new ord -> old ord.
     * @param vectors the float vectors to reorder
     * @param numThreads number of CPU threads to use
     * @return permutation array where permutation[newOrd] = oldOrd
     */
    int[] computePermutation(float[][] vectors, int numThreads);
}
```

### TODO 2: Implement `BipartiteReorderStrategy`

Wrap existing `BpReorderer` behind the new interface.

**File:** `k-NN/src/main/java/org/opensearch/knn/memoryoptsearch/faiss/reorder/bpreorder/BipartiteReorderStrategy.java`

- Delegates to existing `BpReorderer.computePermutation()`
- Passes through `numThreads` to the BP algorithm's `ForkJoinPool`

### TODO 3: Implement `KMeansReorderStrategy`

Wrap existing `ClusterSorter` behind the new interface.

**File:** `k-NN/src/main/java/org/opensearch/knn/memoryoptsearch/faiss/reorder/kmeansreorder/KMeansReorderStrategy.java`

- Delegates to existing `ClusterSorter.clusterAndSort()`
- Uses `FaissKMeansService` (JNI) when available, falls back to `KMeansClusterer` (pure Java)

### TODO 4: Create `SegmentReorderService`

Orchestrator that decides whether to reorder and applies the reorder to `.vec` and `.faiss` files.

**File:** `k-NN/src/main/java/org/opensearch/knn/memoryoptsearch/faiss/reorder/SegmentReorderService.java`

Responsibilities:
- Check if segment size > 10k vectors threshold (configurable constant)
- Extract vectors from the just-written `.vec` file (or from the in-memory buffer during flush)
- Call `VectorReorderStrategy.computePermutation()`
- Build `ReorderOrdMap` from the permutation
- Rewrite `.vec` file in reordered order using `ReorderedFlatVectorsWriter`
- Rewrite `.faiss` file using `FaissIndexReorderTransformer` + the appropriate `FaissIndexReorderer` (HNSW, flat, etc.)
- Write the docid-to-ord skip list index for reordered access

Key constants:
```java
private static final int MIN_VECTORS_FOR_REORDER = 10_000;
private static final int DEFAULT_REORDER_THREADS = 4; // configurable
```

### TODO 5: Integrate into flush path

**File to modify:** `NativeEngines990KnnVectorsWriter.flush()`

After the existing `writer.flushIndex()` call completes (which writes `.vec` and `.faiss`), add:

```java
// After writer.flushIndex() succeeds:
if (totalLiveDocs >= SegmentReorderService.MIN_VECTORS_FOR_REORDER) {
    SegmentReorderService reorderService = new SegmentReorderService(
        segmentWriteState, fieldInfo, reorderStrategy
    );
    reorderService.reorderSegmentFiles();
}
```

Alternatively (preferred approach): integrate reordering INSIDE the flush, reordering the in-memory vectors before they are written. This avoids a read-rewrite cycle:
- After `flatVectorsWriter.flush()` writes the `.vec`, but before `writer.flushIndex()` builds the `.faiss`:
  1. Read vectors from the in-memory `field.getVectors()` map
  2. Compute permutation via `VectorReorderStrategy`
  3. Create a reordered `KNNVectorValues` wrapper that yields vectors in permuted order
  4. Pass the reordered values to `writer.flushIndex()`
  5. Also rewrite the `.vec` in reordered order (or flush it in reordered order from the start)

Decision: The second approach (reorder before writing) is cleaner but requires modifying how `flatVectorsWriter.flush()` works. The first approach (rewrite after writing) is simpler to implement and uses existing `ReorderedFlatVectorsWriter` + `FaissIndexReorderTransformer` infrastructure. **Start with the post-write rewrite approach.**

### TODO 6: Integrate into merge path

**File to modify:** `NativeEngines990KnnVectorsWriter.mergeOneField()`

Same pattern as flush: after `writer.mergeIndex()` completes, apply reordering if the merged segment exceeds the threshold.

```java
// After writer.mergeIndex() succeeds:
if (totalLiveDocs >= SegmentReorderService.MIN_VECTORS_FOR_REORDER) {
    SegmentReorderService reorderService = new SegmentReorderService(
        segmentWriteState, fieldInfo, reorderStrategy
    );
    reorderService.reorderSegmentFiles();
}
```

### TODO 7: Wire up strategy selection

The `VectorReorderStrategy` instance needs to be passed through the codec stack:

1. `NativeEngines990KnnVectorsFormat` constructor → accepts strategy (or strategy name)
2. `NativeEngines990KnnVectorsWriter` constructor → receives strategy
3. `flush()` and `mergeOneField()` → use strategy

For now, default to `BipartiteReorderStrategy`. Could later be configurable via index settings:
```
index.knn.reorder.strategy = "bipartite" | "kmeans" | "none"
index.knn.reorder.min_vectors = 10000
index.knn.reorder.threads = 4
```

### TODO 8: Handle `.faiss` file reordering

The existing reorder infrastructure already handles this via:
- `FaissIndexReorderTransformer` - reads `.faiss`, applies permutation, writes reordered `.faiss`
- `IndexTypeToFaissIndexReordererMapping` - maps FAISS index types to their reorderers
- `FaissHNSWIndexReorderer` / `FaissHnswReorderer` - reorders HNSW graph neighbor lists
- `FaissIndexFloatFlatReorderer` / `FaissIndexBinaryFlatReorderer` - reorders flat index data
- `FaissIdMapIndexReorderer` - reorders ID map wrapper

These work on files. During flush/merge, we need to:
1. Let the normal flush/merge write the `.faiss` file
2. Read it back, apply the reorder permutation, write a new reordered `.faiss`
3. Replace the original with the reordered version (atomic rename)

### TODO 9: Handle `.vec` file reordering

Use existing `ReorderedFlatVectorsWriter` which:
- Reads vectors from the original `.vec` via `ReorderedLucene99FlatVectorsReader`
- Writes them in permuted order to a new `.vec` file
- Handles the Lucene codec header/footer correctly

### TODO 10: Thread pool for reordering

Reordering (especially BP) is CPU-intensive. Options:
- Use a dedicated `ThreadPool.Names.KNN_REORDER` thread pool (new)
- Use the existing `GENERIC` thread pool
- Use `ForkJoinPool` with configurable parallelism (current BP approach)

For now: use `ForkJoinPool` with `DEFAULT_REORDER_THREADS` parallelism, matching the existing BP implementation. The thread count should be a constant that can be changed later to a setting.

---

## Test Plan

### Unit Tests

**TODO 11: `VectorReorderStrategyTests`**
- Test `BipartiteReorderStrategy.computePermutation()` produces a valid permutation (all ords present, no duplicates)
- Test `KMeansReorderStrategy.computePermutation()` produces a valid permutation
- Test with edge cases: exactly 10k vectors, 10001 vectors, 1 vector, 0 vectors
- Test that permutation improves locality (vectors in same cluster are adjacent)

**TODO 12: `SegmentReorderServiceTests`**
- Test that segments below 10k threshold are NOT reordered
- Test that segments at/above 10k threshold ARE reordered
- Test that reordered `.vec` file contains same vectors in different order
- Test that reordered `.faiss` file is valid and searchable
- Test that the docid-to-ord mapping is consistent after reorder

**TODO 13: `NativeEngines990KnnVectorsWriterReorderTests`**
- Extend existing `NativeEngines990KnnVectorsWriterFlushTests`:
  - Add test case: flush with >10k vectors → verify reorder was applied
  - Add test case: flush with <10k vectors → verify reorder was NOT applied
- Extend existing `NativeEngines990KnnVectorsWriterMergeTests`:
  - Add test case: merge producing >10k vectors → verify reorder was applied

### Integration Tests

**TODO 14: End-to-end reorder during indexing**
- Index >10k vectors into a k-NN index
- Force flush
- Verify the segment's `.vec` file is reordered (vectors are not in insertion order)
- Run k-NN search queries and verify correct results (recall is maintained)
- Compare search latency with and without reordering (reordered should be faster or equal)

**TODO 15: End-to-end reorder during force merge**
- Index vectors across multiple flushes (creating multiple segments)
- Run `_forcemerge?max_num_segments=1`
- Verify the merged segment is reordered
- Verify search correctness

**TODO 16: Strategy pluggability test**
- Configure index with BP reorder strategy → verify BP is used
- Configure index with KMeans reorder strategy → verify KMeans is used
- Configure index with "none" → verify no reordering

### Performance Tests

**TODO 17: Reorder overhead benchmarks**
- Measure flush latency with and without reordering for various segment sizes (10k, 100k, 500k, 1M vectors)
- Measure merge latency with and without reordering
- Measure search latency improvement from reordering (p50, p99)
- Use existing OSB benchmark infrastructure (`do_reorder_bp.sh` pattern)

### Correctness Tests

**TODO 18: Verify search recall is preserved**
- Index a known dataset (e.g., sift-1M subset)
- Run exact k-NN search (brute force) to get ground truth
- Run approximate k-NN search on reordered index
- Verify recall@10 is identical to non-reordered index (reordering should not affect recall, only latency)

---

## File Change Summary

| File | Action | Description |
|------|--------|-------------|
| `VectorReorderStrategy.java` | NEW | Strategy interface for pluggable reordering |
| `BipartiteReorderStrategy.java` | NEW | BP implementation of strategy |
| `KMeansReorderStrategy.java` | NEW | KMeans implementation of strategy |
| `SegmentReorderService.java` | NEW | Orchestrator: threshold check, permutation, file rewriting |
| `NativeEngines990KnnVectorsWriter.java` | MODIFY | Add reorder calls after flush and merge |
| `NativeEngines990KnnVectorsFormat.java` | MODIFY | Accept/pass reorder strategy |
| `NativeEngines990KnnVectorsWriterFlushTests.java` | MODIFY | Add reorder test cases |
| `NativeEngines990KnnVectorsWriterMergeTests.java` | MODIFY | Add reorder test cases |
| New integration test class | NEW | End-to-end reorder tests |
