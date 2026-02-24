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

**Both flush and merge paths**, post-write reorder with mmap-backed vector access.

**Approach: Post-write reorder using mmap'd `FloatVectorValues` from `Lucene99FlatVectorsReader`**

After both `.vec` and `.faiss` files are written by the normal flush/merge pipeline:
1. Open the `.vec` file via `Lucene99FlatVectorsReader` → returns mmap-backed `FloatVectorValues`
2. Pass `FloatVectorValues` directly to `BpVectorReorderer.computeValueMap()` — vectors are
   accessed via `vectorValue(int ord)` random access on the mmap'd file, never copied to heap
3. Each BP worker thread calls `vectors.copy()` which creates another mmap view (see
   `BpVectorReorderer.PerThreadState`, line ~107 of `BpVectorReorderer.java`)
4. Only heap cost: `2 * 4 * numVectors` bytes for `sortedIds[]` + `biases[]` arrays
   (validated by `BpVectorReorderer.docRAMRequirements()`)
5. Apply permutation: rewrite `.vec` via `ReorderedFlatVectorsWriter`, rewrite `.faiss` via
   `FaissIndexReorderTransformer`

**Why this works for both flush and merge:**
- Flush: `flatVectorsWriter.flush()` writes `.vec` → `NativeIndexWriter.flushIndex()` writes `.faiss` → reorder both
- Merge: `flatVectorsWriter.mergeOneField()` writes merged `.vec` → `NativeIndexWriter.mergeIndex()` writes `.faiss` → reorder both
- In both cases the `.vec` file exists on disk before reorder runs, so mmap works identically

**Why NOT pre-write reorder:**
- `flatVectorsWriter.flush()` writes `.vec` from the in-memory buffer — intercepting this requires
  modifying Lucene's `FlatVectorsWriter` contract
- The FAISS graph build (`MemOptimizedNativeIndexBuildStrategy`) inserts vectors with their doc IDs
  via `JNIService.insertToIndex(docIds, vectorAddress, ...)` — reordering the input iterator would
  change the doc-id-to-vector mapping, requiring careful coordination
- Post-write reorder is simpler and uses existing `ReorderedFlatVectorsWriter` + `FaissIndexReorderTransformer`

**Not recommended: Refresh path**
- Refresh does not write segment files; it only opens readers
- No codec writer methods are invoked during refresh
- Would require a fundamentally different architecture (post-hoc rewriting)

---

## Implementation Plan

### TODO 1: Define `VectorReorderStrategy` interface (pluggable reordering)

Create a strategy interface so BP and KMeans (and future strategies) are interchangeable.
The interface accepts `FloatVectorValues` (not `float[][]`) so implementations can work with
mmap-backed vectors without loading everything into heap.

**File:** `k-NN/src/main/java/org/opensearch/knn/memoryoptsearch/faiss/reorder/VectorReorderStrategy.java`

```java
public interface VectorReorderStrategy {
    /**
     * Compute a permutation array mapping new ord -> old ord.
     * @param vectors FloatVectorValues — may be mmap-backed (from Lucene99FlatVectorsReader)
     *                or heap-backed (from FloatVectorValues.fromFloats())
     * @param numThreads number of CPU threads to use
     * @return permutation array where permutation[newOrd] = oldOrd
     */
    int[] computePermutation(FloatVectorValues vectors, int numThreads);
}
```

### TODO 2: Implement `BipartiteReorderStrategy` (mmap-friendly)

Wrap Lucene's `BpVectorReorderer` behind the new interface. The key insight is that
`BpVectorReorderer.computeValueMap()` accepts `FloatVectorValues` directly — it does NOT
require all vectors in a `float[][]` on heap. It accesses vectors via random-access
`vectorValue(int ord)` calls, and each thread gets its own `vectors.copy()` (another mmap view).

**Memory overhead:** Only `2 * 4 bytes * numVectors` for the `sortedIds` + `biases` arrays
(checked by `BpVectorReorderer.docRAMRequirements()`). The vectors themselves stay on disk
via mmap.

**File:** `k-NN/src/main/java/org/opensearch/knn/memoryoptsearch/faiss/reorder/bpreorder/BipartiteReorderStrategy.java`

```java
public class BipartiteReorderStrategy implements VectorReorderStrategy {
    @Override
    public int[] computePermutation(FloatVectorValues vectors, int numThreads) {
        // BpVectorReorderer reads vectors via vectorValue(ord) — works with mmap-backed FloatVectorValues
        // See: k-NN/src/main/java/org/apache/lucene/misc/index/BpVectorReorderer.java
        //   - PerThreadState calls vectors.copy() per thread (line ~107) — each gets own mmap view
        //   - Hot loop: vectors.vectorValue(ids[i]) (line ~370 in ComputeBiasTask.compute())
        //   - RAM check: docRAMRequirements() only accounts for int[] + float[] metadata, not vectors
        BpVectorReorderer reorderer = new BpVectorReorderer("vectors");
        reorderer.setMinPartitionSize(1);
        ForkJoinPool pool = new ForkJoinPool(numThreads);
        try {
            TaskExecutor executor = new TaskExecutor(pool);
            Sorter.DocMap map = reorderer.computeValueMap(
                vectors, VectorSimilarityFunction.EUCLIDEAN, executor
            );
            int[] permutation = new int[vectors.size()];
            for (int i = 0; i < permutation.length; i++) {
                permutation[i] = map.newToOld(i);
            }
            return permutation;
        } finally {
            pool.shutdown();
        }
    }
}
```

**Contrast with current `BpReorderer` (heap-based):**
The existing `bpreorder/BpReorderer.java` forces all vectors into heap:
```java
// Current approach — loads ALL vectors into Java heap:
FloatVectorValues fvv = FloatVectorValues.fromFloats(Arrays.asList(vectors), dim);
```
The new strategy avoids this by passing the `FloatVectorValues` from `Lucene99FlatVectorsReader`
directly, which is backed by mmap'd `.vec` file on disk.

### TODO 3: Implement `KMeansReorderStrategy`

Wrap existing `ClusterSorter` behind the new interface.

**File:** `k-NN/src/main/java/org/opensearch/knn/memoryoptsearch/faiss/reorder/kmeansreorder/KMeansReorderStrategy.java`

- Delegates to existing `ClusterSorter.clusterAndSort()`
- Uses `FaissKMeansService` (JNI) when available, falls back to `KMeansClusterer` (pure Java)
- Note: KMeans currently requires `float[][]` in heap (JNI and pure-Java both need contiguous
  vector data). Future optimization: stream vectors through JNI in batches.

### TODO 4: Create `SegmentReorderService`

Orchestrator that decides whether to reorder and applies the reorder to `.vec` and `.faiss` files.

**File:** `k-NN/src/main/java/org/opensearch/knn/memoryoptsearch/faiss/reorder/SegmentReorderService.java`

Responsibilities:
- Check if segment size > 10k vectors threshold (configurable constant)
- Open a `Lucene99FlatVectorsReader` on the just-written `.vec` file to get mmap-backed `FloatVectorValues`
- Call `VectorReorderStrategy.computePermutation(floatVectorValues, numThreads)` — no heap copy for BP
- Build `ReorderOrdMap` from the permutation
- Rewrite `.vec` file in reordered order using `ReorderedFlatVectorsWriter`
- Rewrite `.faiss` file using `FaissIndexReorderTransformer` + the appropriate `FaissIndexReorderer` (HNSW, flat, etc.)
- Write the docid-to-ord skip list index for reordered access

Key constants:
```java
private static final int MIN_VECTORS_FOR_REORDER = 10_000;
private static final int DEFAULT_REORDER_THREADS = 4; // configurable
```

### TODO 5: Integrate into flush path (mmap-based, post-write reorder)

**File to modify:** `NativeEngines990KnnVectorsWriter.flush()`

The integration happens AFTER both `.vec` and `.faiss` are written. The sequence in `flush()` is:

```
flatVectorsWriter.flush(maxDoc, sortMap)   → writes .vec to disk
writer.flushIndex(knnVectorValuesSupplier) → builds FAISS graph, writes .faiss to disk
// ← INSERT REORDER HERE
```

The reorder step:
1. Open the just-written `.vec` via `Lucene99FlatVectorsReader` → get mmap-backed `FloatVectorValues`
2. Pass `FloatVectorValues` to `VectorReorderStrategy.computePermutation()` — BP reads vectors
   via `vectorValue(ord)` from mmap, never loads all vectors into heap
3. Build `ReorderOrdMap` from permutation
4. Rewrite `.vec` in reordered order via `ReorderedFlatVectorsWriter` (reads from mmap reader, writes new file)
5. Rewrite `.faiss` via `FaissIndexReorderTransformer` (remaps HNSW neighbor lists)
6. Atomic rename new files over originals

```java
// In NativeEngines990KnnVectorsWriter.flush(), after writer.flushIndex():
if (totalLiveDocs >= SegmentReorderService.MIN_VECTORS_FOR_REORDER) {
    // Opens .vec via Lucene99FlatVectorsReader (mmap-backed FloatVectorValues)
    // BP only needs 2*4*N bytes heap for metadata (sortedIds + biases arrays)
    // Vectors stay on disk, accessed via vectorValue(ord) random access
    SegmentReorderService reorderService = new SegmentReorderService(
        segmentWriteState, fieldInfo, reorderStrategy
    );
    reorderService.reorderSegmentFiles();
}
```

**Why post-write reorder (not pre-write):**
- `flatVectorsWriter.flush()` writes `.vec` from the in-memory buffer — we can't easily intercept this
- The `.faiss` graph is built by `MemOptimizedNativeIndexBuildStrategy.buildAndWriteIndex()` which
  streams vectors via `KNNVectorValues` iterator and inserts into FAISS via JNI (`JNIService.insertToIndex()`)
  — reordering the iterator would change doc-id-to-vector mapping in the graph
- Post-write reorder is simpler: rewrite both files with consistent permutation using existing
  `ReorderedFlatVectorsWriter` + `FaissIndexReorderTransformer` infrastructure

**Memory profile during reorder (BP, 1M vectors, 128-dim):**
- Vectors: 0 bytes heap (mmap from `.vec` file, ~512MB on disk)
- BP metadata: `2 * 4 * 1M` = ~8MB heap (sortedIds + biases)
- Per-thread state: `3 * 128 * 4` bytes per thread (centroids + scratch) = negligible
- Total heap: ~8MB regardless of vector count or dimension

### TODO 6: Integrate into merge path (mmap-based, post-write reorder)

**File to modify:** `NativeEngines990KnnVectorsWriter.mergeOneField()`

Same pattern as flush. The merge sequence is:

```
flatVectorsWriter.mergeOneField(fieldInfo, mergeState)  → writes merged .vec (streams from source segments)
writer.mergeIndex(knnVectorValuesSupplier, totalLiveDocs) → builds FAISS graph, writes .faiss
// ← INSERT REORDER HERE
```

During merge, source segment vectors are NOT loaded into heap — they're read via
`MergedVectorValues.mergeFloatVectorValues()` which is a lazy iterator over the source
segments' on-disk `.vec` files. After the merged `.vec` is written, we open it with
`Lucene99FlatVectorsReader` (mmap) and pass to BP — same zero-heap-copy approach as flush.

```java
// In NativeEngines990KnnVectorsWriter.mergeOneField(), after writer.mergeIndex():
if (totalLiveDocs >= SegmentReorderService.MIN_VECTORS_FOR_REORDER) {
    SegmentReorderService reorderService = new SegmentReorderService(
        segmentWriteState, fieldInfo, reorderStrategy
    );
    reorderService.reorderSegmentFiles();
}
```

**Force merge note:** A `_forcemerge?max_num_segments=1` on a 10M vector index will:
1. Stream all vectors from source segments → write merged `.vec` (disk-to-disk, no heap)
2. Build FAISS HNSW graph in native memory (this IS fully in native memory)
3. Reorder: open merged `.vec` via mmap, compute BP permutation (~80MB heap for metadata),
   rewrite `.vec` and `.faiss` with permutation

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

**TODO 11a: `BpReordererTests`** ✅ DONE
- Test `BpReorderer.computePermutation(float[][])` produces a valid permutation (all ords present, no duplicates)
- Test with small input (2 vectors), identical vectors
- Test that BP groups well-separated clusters (>90% same-cluster adjacency)

**TODO 11b: `BipartiteReorderStrategyTests`**
- Test `BipartiteReorderStrategy.computePermutation(FloatVectorValues, numThreads)` — the new mmap-friendly interface
- Create `FloatVectorValues` via `FloatVectorValues.fromFloats()` (heap-backed, simulates mmap contract)
- Verify valid permutation (all ords present, no duplicates)
- Verify clustering quality: interleaved 2-cluster input → >90% same-cluster adjacency after reorder
- Test with `numThreads=1` and `numThreads=4` — both produce valid permutations
- Edge case: exactly 2 vectors (minimum for BP)

**TODO 11c: `KMeansReorderStrategyTests`**
- Test `KMeansReorderStrategy.computePermutation(FloatVectorValues, numThreads)` — the new interface
- Create `FloatVectorValues` via `FloatVectorValues.fromFloats()`
- Verify valid permutation (all ords present, no duplicates)
- Verify clustering quality: interleaved 3-cluster input → >80% same-cluster adjacency
- Test with custom k and niter parameters
- Edge case: k > numVectors (should cap k at numVectors)
- Must set `kmeans.useJni=false` to avoid JNI dependency in unit tests

**TODO 11d: `ClusterSorterTests`** ✅ DONE
- Test `ClusterSorter.sortByCluster()` groups by cluster, sorts by distance within cluster
- Test `KMeansClusterer.cluster()` produces valid assignments and non-negative distances
- Test `ClusterSorter.clusterAndSort()` end-to-end with pure Java path

**TODO 11e: `ReorderOrdMapTests`** ✅ DONE
- Test newOrd2Old → oldOrd2New inversion
- Test round-trip: `oldOrd2New[newOrd2Old[i]] == i`
- Edge cases: single element, identity, reverse

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

**TODO 19: `SegmentReorderService` integration test (manual / `./gradlew run`)**

This tests the full orchestration: flush/merge → reorder → search. Requires a running OpenSearch
instance with the k-NN plugin because `SegmentReorderService` depends on real Lucene segment files
(`.vec`, `.faiss`, `.vemf`) written by the codec.

**Setup:**
```bash
# Copy non-reordered segments before starting
cp -r /Users/finnrobl/Documents/k-NN-2/index-backups/* /Users/finnrobl/Documents/k-NN-2/e2e_data/

# Start OpenSearch with k-NN plugin
cd /Users/finnrobl/Documents/k-NN-2/k-NN
./gradlew run -Ddata=/Users/finnrobl/Documents/k-NN-2/e2e_data
```

**Test steps:**
1. Create a k-NN index with reorder enabled (BP strategy)
2. Index >10k vectors (e.g., 50k from sift dataset)
3. Call `_flush` → triggers `NativeEngines990KnnVectorsWriter.flush()` → `maybeReorderSegmentFiles()`
4. Verify segment files were rewritten:
   - `.vec` file: open with `Lucene99FlatVectorsReader`, verify vectors are NOT in insertion order
   - `.faiss` file: load with `FaissIndex.load()`, verify HNSW graph neighbor lists reference valid ords
5. Run k-NN search queries, verify results match non-reordered baseline (same recall@10)
6. Call `_forcemerge?max_num_segments=1` → triggers `mergeOneField()` → reorder on merged segment
7. Repeat verification on merged segment

**Validation checks:**
- `ReorderOrdMap` round-trip: for every vector, `oldOrd2New[newOrd2Old[i]] == i`
- Vector content preserved: every vector in the reordered `.vec` exists in the original (just different order)
- FAISS graph valid: all neighbor IDs in HNSW graph are in range `[0, numVectors)`
- Search correctness: recall@10 identical to non-reordered index
- No orphaned `.reorder` temp files left in segment directory

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
