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

## Recommended Integration Point (RESOLVED 2026-02-25)

**Merge path only**, via `NativeEngines990KnnVectorsWriter.finish()`.

**Approach: Post-write reorder in `finish()` after `.vec`/`.vemf` footers are flushed to disk**

The reorder runs inside `finish()`, which is called by `KnnVectorsWriter.merge()` after all
`mergeOneField()` calls complete. The sequence:

1. `mergeOneField()` writes `.vec` data and `.faiss` normally, marks fields >= 10k vectors
   in `fieldsToReorder`
2. `finish()` calls `flatVectorsWriter.finish()` → writes `.vec`/`.vemf` codec footers
3. `finish()` calls `flatVectorsWriter.close()` → flushes `IndexOutput` buffers to disk
4. For each field in `fieldsToReorder`, `SegmentReorderService.reorderSegmentFiles()`:
   a. Opens finalized `.vec` via `Lucene99FlatVectorsReader` (mmap-backed `FloatVectorValues`)
   b. Computes BP permutation — zero-heap-copy, only `2*4*N` bytes for metadata arrays
   c. Rewrites `.vec` + `.vemf` via `ReorderedFlatVectorsWriter` (reordered layout with skip list)
   d. Rewrites `.faiss` via `FaissIndexReorderTransformer` (remapped HNSW neighbor lists)
   e. Atomic rename of reordered files over originals

**Why `finish()` and not `mergeOneField()`:**
During `mergeOneField()`, the `.vec`/`.vemf` files are open `IndexOutput` streams without
finalized codec footers. `Lucene99FlatVectorsReader` validates footers on open via
`CodecUtil.checkFooter()` / `CodecUtil.retrieveChecksum()`, failing with
`CorruptIndexException: checksum status indeterminate: remaining=0`.
Footers are only written by `Lucene99FlatVectorsWriter.finish()`. Additionally,
`IndexOutput` buffers must be flushed via `close()` before the data is readable on disk.

**Why not `KNN80CompoundFormat.write()`:**
Compound file creation is skipped for large force-merged segments (`compound=False`).
`finish()` runs for ALL merges regardless of compound file creation.

**Why not flush path:**
Flush segments are typically small. They get merged later, at which point reordering applies.
The `fieldsToReorder` list is only populated by `mergeOneField()`, not `flush()`.

**Why not refresh path:**
Refresh does not write segment files; it only opens readers.

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

### TODO 5: ~~Integrate into flush path~~ SKIPPED

Flush is NOT reordered. Flush segments are typically small and will be merged later.
The `fieldsToReorder` list is only populated by `mergeOneField()`, not `flush()`.

### TODO 6: Integrate into merge path via `finish()` ✅ DONE

**File modified:** `NativeEngines990KnnVectorsWriter.java`

The reorder cannot happen inside `mergeOneField()` because `.vec`/`.vemf` files don't have
finalized codec footers yet (see Error 1 in Manual Testing Results). Instead:

1. `mergeOneField()` marks fields >= 10k vectors in `fieldsToReorder`
2. `finish()` calls `flatVectorsWriter.finish()` → writes `.vec`/`.vemf` footers
3. `finish()` calls `flatVectorsWriter.close()` → flushes `IndexOutput` buffers to disk
4. `finish()` iterates `fieldsToReorder` and calls `SegmentReorderService.reorderSegmentFiles()`
   for each, which rewrites `.vec` + `.vemf` + `.faiss`

```java
// In NativeEngines990KnnVectorsWriter.mergeOneField(), at the end:
if (reorderStrategy != null && totalLiveDocs >= SegmentReorderService.MIN_VECTORS_FOR_REORDER) {
    fieldsToReorder.add(fieldInfo);
}

// In NativeEngines990KnnVectorsWriter.finish():
flatVectorsWriter.finish();  // writes footers
flatVectorsWriter.close();   // flushes to disk
for (FieldInfo fieldInfo : fieldsToReorder) {
    SegmentReorderService reorderService = new SegmentReorderService(
        segmentWriteState, fieldInfo, reorderStrategy
    );
    reorderService.reorderSegmentFiles();  // rewrites .vec + .vemf + .faiss
}
```

**Call chain during merge:**
```
SegmentMerger.mergeVectorValues()
  → try (KnnVectorsWriter writer = codec.fieldsWriter(state)) {
        writer.merge(mergeState);
     }
     // merge() internally:
     //   1. mergeOneField() per field  → writes .vec data, .faiss; marks fieldsToReorder
     //   2. finish()                   → writes .vec/.vemf footers
     //                                 → closes flatVectorsWriter (flushes to disk)
     //                                 → reorders .vec + .vemf + .faiss  ← HERE
     // try-with-resources:
     //   3. close()                    → cleanup (flatVectorsWriter already closed)
```

**Force merge note:** A `_forcemerge?max_num_segments=1` on a 10M vector index will:
1. Stream all vectors from source segments → write merged `.vec` (disk-to-disk, no heap)
2. Build FAISS HNSW graph in native memory
3. `finish()` writes footers, closes streams
4. Reorder: open merged `.vec` via mmap, compute BP permutation (~80MB heap for metadata),
   rewrite `.vec`, `.vemf`, and `.faiss` with permutation

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
- Extend existing `NativeEngines990KnnVectorsWriterMergeTests`:
  - Add test case: merge producing >10k vectors → verify reorder was applied (`.vec` rewritten
    with `ReorderedLucene99FlatVectorsReader111` codec, `.faiss` neighbor lists remapped)
  - Add test case: merge producing <10k vectors → verify reorder was NOT applied
  - Add test case: merge with `reorderStrategy=null` → verify reorder was NOT applied
- Flush tests: verify flush does NOT trigger reorder regardless of vector count

### Integration Tests

**TODO 14: End-to-end reorder during force merge**
- Index >10k vectors into a k-NN index across multiple flushes
- Run `_forcemerge?max_num_segments=1`
- Verify the merged segment's `.vec` file is reordered (uses `ReorderedLucene99FlatVectorsReader111` codec)
- Verify the merged segment's `.faiss` file has remapped neighbor lists
- Run k-NN search queries and verify correct results (recall is maintained)
- Verify reorder works even when final segment has `compound=False`

**TODO 15: End-to-end reorder during background merge**
- Index vectors continuously to trigger background merges via `TieredMergePolicy`
- Verify that merged segments above 10k vectors are reordered
- Verify that merged segments below 10k vectors are NOT reordered
- Verify search correctness throughout

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
3. Call `_forcemerge?max_num_segments=1` → triggers `mergeOneField()` → marks field → `finish()` reorders
4. Verify segment files were rewritten:
   - `.vec` file: open with `ReorderedLucene99FlatVectorsReader111`, verify vectors accessible via ordMap
   - `.faiss` file: load with `FaissIndex.load()`, verify HNSW graph neighbor lists reference valid ords
5. Run k-NN search queries, verify results match non-reordered baseline (same recall@10)
6. Verify that the final force-merged segment IS reordered even when `compound=False`

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

## Manual Testing Results (2026-02-24)

### Error 1: Cannot read `.vec`/`.vemf` during flush — files not yet finalized

**When:** `SegmentReorderService.computePermutationFromVecFile()` tries to open the `.vemf` file
with `Lucene99FlatVectorsReader` during `flush()`.

**Error:**
```
org.apache.lucene.index.CorruptIndexException: checksum status indeterminate: remaining=0;
please run checkindex for more details
(resource=BufferedChecksumIndexInput(MemorySegmentIndexInput(
  path=".../_0_NativeEngines990KnnVectorsFormat_0.vemf")))
    at org.apache.lucene.codecs.CodecUtil.checkFooter(CodecUtil.java:483)
    at org.apache.lucene.codecs.lucene99.Lucene99FlatVectorsReader.readMetadata(Lucene99FlatVectorsReader.java:112)
    at org.apache.lucene.codecs.lucene99.Lucene99FlatVectorsReader.<init>(Lucene99FlatVectorsReader.java:69)
    at o.o.knn.memoryoptsearch.faiss.reorder.SegmentReorderService.computePermutationFromVecFile(SegmentReorderService.java:126)
    at o.o.knn.memoryoptsearch.faiss.reorder.SegmentReorderService.reorderSegmentFiles(SegmentReorderService.java:87)
    at o.o.knn.index.codec.KNN990Codec.NativeEngines990KnnVectorsWriter.maybeReorderSegmentFiles(NativeEngines990KnnVectorsWriter.java:158)
    at o.o.knn.index.codec.KNN990Codec.NativeEngines990KnnVectorsWriter.flush(NativeEngines990KnnVectorsWriter.java:148)
    at org.apache.lucene.codecs.perfield.PerFieldKnnVectorsFormat$FieldsWriter.flush(PerFieldKnnVectorsFormat.java:120)
    at org.apache.lucene.index.VectorValuesConsumer.flush(VectorValuesConsumer.java:76)
    at org.apache.lucene.index.IndexingChain.flush(IndexingChain.java:305)
```

**Root cause:** During `flush()`, the `.vec` and `.vemf` files are written by `flatVectorsWriter.flush()`
but their codec footer has not been written yet (Lucene writes the footer later in the flush pipeline).
`Lucene99FlatVectorsReader` validates the footer checksum on open, which fails because the file is
incomplete.

**Additionally:** Even if we could read the files, writing `.reorder` temp files into the segment
directory causes Lucene's compound file writer (`Lucene90CompoundFormat.write()`) to pick them up
and fail with:
```
CorruptIndexException: compound sub-files must have a valid codec header and footer:
file is too small (0 bytes) (resource=...vemf.reorder)
```

### Manual Test: Merge Path (2026-02-24 13:21)

**Setup:** 15k vectors in 3 clusters (interleaved), 3 segments, force merge to 1.

**Result:** Force merge succeeded. BP permutation computed in 187ms. Search returns results without errors.

**Problem:** Pre-write reorder approach (remapping vectors in `KNNVectorValues` supplier) breaks
the doc-ID-to-vector correspondence. Search returns wrong doc IDs because the FAISS graph stores
reordered vectors under original doc IDs.

**Correct approach (from `ReorderAllWithBP` script):**
The original script rewrites BOTH `.vec` and `.faiss` files post-write:
1. Read vectors from finalized `.vec` via `Lucene99FlatVectorsReader`
2. Compute BP permutation
3. Rewrite `.vec` via `ReorderedFlatVectorsWriter` — vectors stored in BP order, new codec header
   (`ReorderedLucene99FlatVectorsFormatMeta`) with ordMap baked into `.vemf` metadata
4. Rewrite `.faiss` via `FaissIndexReorderTransformer` — HNSW neighbor lists remapped
5. Replace original files

The contract:
- Doc IDs in FAISS graph: **unchanged**
- HNSW neighbor lists: **remapped** to point to reordered ords
- `.vec` vector storage: **reordered** (BP order)
- `.vemf` metadata: **new codec** with ordMap so `ReorderedLucene99FlatVectorsReader111` can translate

**Integration point for merge: `KNN80CompoundFormat.write()`**

This is the right hook because:
- It runs AFTER `SegmentMerger.merge()` completes (all `mergeOneField()` calls done, files finalized with footers)
- It runs BEFORE the compound file is created
- It already iterates over engine files (`.faiss`) and copies them out of the compound file
- We can insert the reorder step here: rewrite `.vec`/`.vemf`/`.faiss` in-place before compound file creation

Lifecycle:
```
SegmentMerger.merge()
  → mergeOneField() per field  → writes .vec, .vemf, .faiss (all finalized)
IndexWriter.createCompoundFile()
  → KNN80CompoundFormat.write()  ← INSERT REORDER HERE (files are finalized, not yet compounded)
    → copies .faiss → .faissc
    → delegate writes remaining files into .cfs
```

### Issue: `KNN80CompoundFormat.write()` not always called (2026-02-24 13:46)

**Observation:** After `_forcemerge?max_num_segments=1` with 15k vectors:
- `_0` (flush, ~10k): not reordered → falls back to standard reader ✅
- `_1` (intermediate merge, 14245): reordered in `KNN80CompoundFormat.write()` ✅
- `_2` (flush, ~755): not reordered, below threshold ✅
- `_3` (final forcemerge of `_1`+`_2`, 15000): **NOT reordered** ❌

Segment `_3` has `compound=False` — Lucene skipped compound file creation for the final
force-merged segment. `KNN80CompoundFormat.write()` was never called for `_3`.

**Root cause:** Lucene's `IndexWriter` only creates compound files when `MergePolicy.useCompoundFile()`
returns true. For force merge, the final segment may not be compounded (depends on segment size
and merge policy settings).

**Resolution options:**
1. Move reorder back into `mergeOneField()` but use the post-write approach — after
   `writer.mergeIndex()` writes the `.faiss`, rewrite it in place. The `.vec`/`.vemf` can also
   be rewritten since the files are finalized at that point (footers written by
   `flatVectorsWriter.mergeOneField()`). Need to verify that `.vec`/`.vemf` footers are written
   by the time `mergeOneField()` returns.

2. Keep reorder in `KNN80CompoundFormat.write()` AND also add it to a post-merge hook. The
   `IndexWriter` calls `SegmentMerger.merge()` → then optionally `createCompoundFile()`. We need
   a hook that runs after merge regardless of compound file creation.

3. Use both hooks: `KNN80CompoundFormat.write()` for segments that get compounded, and
   `NativeEngines990KnnVectorsWriter.finish()` or a merge listener for non-compounded segments.

### Error 2: `AssertionError: got context=DEFAULT` during merge reorder (2026-02-25 11:33)

**When:** `SegmentReorderService.rewriteVecFile()` calls `ReorderedFlatVectorsWriter` constructor,
which calls `directory.createOutput(metaFileName, IOContext.DEFAULT)`.

**Error:**
```
java.lang.AssertionError: got context=DEFAULT
    at org.apache.lucene.index.ConcurrentMergeScheduler$1.createOutput(ConcurrentMergeScheduler.java:307)
    at org.apache.lucene.store.TrackingDirectoryWrapper.createOutput(TrackingDirectoryWrapper.java:41)
    at o.o.knn.memoryoptsearch.faiss.reorder.ReorderedFlatVectorsWriter.<init>(ReorderedFlatVectorsWriter.java:46)
    at o.o.knn.memoryoptsearch.faiss.reorder.SegmentReorderService.rewriteVecFile(SegmentReorderService.java:141)
    at o.o.knn.memoryoptsearch.faiss.reorder.SegmentReorderService.reorderSegmentFiles(SegmentReorderService.java:78)
    at o.o.knn.index.codec.KNN990Codec.NativeEngines990KnnVectorsWriter.finish(NativeEngines990KnnVectorsWriter.java:212)
```

**Root cause:** During merge, Lucene's `ConcurrentMergeScheduler` wraps the segment directory
with a delegate that asserts all `createOutput()` calls use `IOContext.MERGE`, not `IOContext.DEFAULT`.
`ReorderedFlatVectorsWriter` hardcodes `IOContext.DEFAULT` in its constructor (line 46 and 77).

**Resolution:** In `SegmentReorderService.reorderSegmentFiles()`, wrap the directory with a
`FilterDirectory` that overrides `createOutput()` to use `state.context` (which is `IOContext.MERGE`
during merge). This avoids modifying `ReorderedFlatVectorsWriter` itself:

```java
final Directory writeDir = new FilterDirectory(directory) {
    @Override
    public IndexOutput createOutput(String name, IOContext context) throws IOException {
        return in.createOutput(name, state.context);
    }
};
```

The `writeDir` is used for all `createOutput` calls (`.vec.reorder`, `.vemf.reorder`, `.faiss.reorder`),
while the original `directory` is used for reads (`openInput`), deletes, and renames.

### Manual Test: `finish()` reorder with IOContext fix (2026-02-25 11:37)

**Setup:** 15k vectors in 3 clusters (interleaved), 3 segments of 5k each, force merge to 1.

**Test strategy:**
1. Create k-NN index (HNSW/faiss, 128-dim, l2)
2. Index 15k vectors in 3 batches of 5k, flush between each → 3 segments
3. Search before force merge → capture baseline doc IDs and scores
4. `_forcemerge?max_num_segments=1` → triggers `mergeOneField()` → `finish()` → reorder
5. Search after force merge → verify same doc IDs and scores
6. Check logs for:
   - `[Reorder] Marked field [my_vector] for reorder (15000 vectors)` — field was marked
   - `[Reorder] Starting reorder for field [my_vector] in segment _3` — reorder started
   - `[Reorder] Permutation computed for 15000 vectors` — BP ran
   - `[Reorder] Completed reorder for field [my_vector]` — reorder finished
   - `[ReorderedReader] Falling back to standard reader` for flush segments _0, _1, _2
   - NO fallback for merged segment _3 → reordered reader used ✅

### Manual Test Result: `finish()` reorder with IOContext fix (2026-02-25 11:39) ✅ PASSED

**Segments before force merge:**
```
_0: 5000 docs, compound=True  (flush)
_1: 5000 docs, compound=True  (flush)
_2: 5000 docs, compound=True  (flush)
```

**Segments after force merge:**
```
_0: 5000 docs, compound=True   (flush, pending deletion)
_1: 5000 docs, compound=True   (flush, pending deletion)
_2: 5000 docs, compound=True   (flush, pending deletion)
_3: 15000 docs, compound=False (force-merged, REORDERED)
```

**Reorder log output:**
```
[Reorder] Marked field [my_vector] for reorder (15000 vectors)
[Reorder] Starting reorder for field [my_vector] in segment _3
[Reorder] Permutation computed for 15000 vectors, field [my_vector]
[Reorder] Completed reorder for field [my_vector] in 462 ms
```

**Reader log output:**
- Segments _0, _1, _2: `[ReorderedReader] Falling back to standard reader` (expected — flush segments, not reordered)
- Segment _3: `[ReorderedReader] Successfully opened REORDERED .vemf for segment _3` ✅

**Search results — BEFORE force merge:**
```
doc_id=8103,  score=0.048115
doc_id=13890, score=0.047268
doc_id=12219, score=0.047071
doc_id=12927, score=0.046295
doc_id=9924,  score=0.046064
```

**Search results — AFTER force merge (reordered):**
```
doc_id=8103,  score=0.048115
doc_id=13890, score=0.047268
doc_id=12219, score=0.047071
doc_id=12927, score=0.046295
doc_id=9924,  score=0.046064
```

**Verdict:** Doc IDs and scores are identical before and after reorder. Reorder was applied to
the force-merged segment `_3` (which has `compound=False`), confirming the `finish()` approach
works for non-compounded segments. The reordered reader (`ReorderedLucene99FlatVectorsReader111`)
is used for the merged segment, while flush segments correctly fall back to the standard reader.

---

## Design Implications (RESOLVED 2026-02-25)

The post-write reorder approach (rewrite `.vec`/`.faiss` after they're written) **cannot work
during `flush()` or `mergeOneField()`** because:
1. `.vec`/`.vemf` files don't have finalized footers yet → can't open with `Lucene99FlatVectorsReader`
2. `IndexOutput` buffers may not be flushed to disk → data not readable via `openInput()`
3. During flush, any new files in the segment directory get included in the compound file → corruption

**Resolution: Reorder in `finish()` (merge path only)**

The `finish()` method is called by `KnnVectorsWriter.merge()` after all `mergeOneField()` calls.
Inside `finish()`:
1. `flatVectorsWriter.finish()` writes `.vec`/`.vemf` codec footers
2. `flatVectorsWriter.close()` flushes `IndexOutput` buffers to disk
3. Files are now readable via `Lucene99FlatVectorsReader`
4. `SegmentReorderService.reorderSegmentFiles()` rewrites `.vec` + `.vemf` + `.faiss`

This avoids the `KNN80CompoundFormat.write()` approach which fails for non-compounded segments.

---

## Merge Path Reorder Design (RESOLVED 2026-02-25)

### Why `mergeOneField()` doesn't work

During `mergeOneField()`, the `.vec`/`.vemf` `IndexOutput` streams are still open without
finalized codec footers. `Lucene99FlatVectorsReader` validates footers on open, failing with
`CorruptIndexException: checksum status indeterminate: remaining=0`.

The footers are written by `Lucene99FlatVectorsWriter.finish()`, which is called by
`KnnVectorsWriter.merge()` AFTER all `mergeOneField()` calls complete.

### Why `KNN80CompoundFormat.write()` doesn't work

Compound file creation is skipped for large force-merged segments (`compound=False`).
The final segment in `_forcemerge?max_num_segments=1` is often not compounded.

### Solution: `finish()` with early close

The Lucene call chain during merge:
```
KnnVectorsWriter.merge(mergeState)
  → mergeOneField() per field     // writes .vec data + .faiss
  → finish()                      // writes footers
SegmentMerger try-with-resources
  → close()                       // closes streams
```

Our `NativeEngines990KnnVectorsWriter.finish()`:
1. Calls `flatVectorsWriter.finish()` — writes `.vec`/`.vemf` codec footers
2. Calls `flatVectorsWriter.close()` — flushes `IndexOutput` buffers to disk
3. For each field in `fieldsToReorder`: opens finalized `.vec` via `Lucene99FlatVectorsReader`,
   computes permutation, rewrites `.vec` + `.vemf` + `.faiss`

The early `close()` is safe because `finish()` is the last method that writes to the streams.
Our `close()` method handles the already-closed `flatVectorsWriter` gracefully via `IOUtils.close()`.

---

## File Change Summary

| File | Action | Description |
|------|--------|-------------|
| `VectorReorderStrategy.java` | NEW ✅ | Strategy interface for pluggable reordering |
| `BipartiteReorderStrategy.java` | NEW ✅ | BP implementation of strategy |
| `KMeansReorderStrategy.java` | NEW | KMeans implementation of strategy |
| `SegmentReorderService.java` | NEW ✅ | Orchestrator: threshold check, permutation, full `.vec` + `.vemf` + `.faiss` rewriting |
| `NativeEngines990KnnVectorsWriter.java` | MODIFY ✅ | `fieldsToReorder` list in `mergeOneField()`, reorder loop in `finish()` after footer flush |
| `NativeEngines990KnnVectorsFormat.java` | MODIFY ✅ | Accept/pass reorder strategy to writer |
| `BasePerFieldKnnVectorsFormat.java` | MODIFY ✅ | Creates `BipartiteReorderStrategy` and passes to format |
| `KNN80CompoundFormat.java` | MODIFY ✅ | Removed all reorder logic, restored to original compound-file-only handling |
| `NativeEngines990KnnVectorsWriterFlushTests.java` | MODIFY | Add reorder test cases |
| `NativeEngines990KnnVectorsWriterMergeTests.java` | MODIFY | Add reorder test cases |
| New integration test class | NEW | End-to-end reorder tests |


---

## Design Change: Move Reorder from CompoundFormat to mergeOneField (2026-02-25)

### Problem

Reordering was in `KNN80CompoundFormat.write()`, which only runs when Lucene creates compound
files. Large force-merged segments (e.g. `_forcemerge?max_num_segments=1`) skip compound file
creation entirely (`compound=False`), so they were never reordered.

### Solution

Moved reorder to `NativeEngines990KnnVectorsWriter.finish()`, which runs for ALL merges
regardless of compound file creation. The reorder happens after `flatVectorsWriter.finish()`
writes the `.vec`/`.vemf` codec footers and `flatVectorsWriter.close()` flushes them to disk.

### How it works

1. During `mergeOneField()`: if the field has >= 10k vectors and a reorder strategy is set,
   the field is added to `fieldsToReorder`
2. In `finish()`: `flatVectorsWriter.finish()` writes `.vec`/`.vemf` footers, then
   `flatVectorsWriter.close()` flushes the `IndexOutput` buffers to disk
3. For each field in `fieldsToReorder`, `SegmentReorderService.reorderSegmentFiles()`:
   a. Opens the finalized `.vec` via `Lucene99FlatVectorsReader` (mmap-backed)
   b. Computes BP permutation from the `FloatVectorValues`
   c. Rewrites `.vec` + `.vemf` via `ReorderedFlatVectorsWriter` (reordered vector layout
      with doc-to-ord skip list in metadata)
   d. Rewrites `.faiss` via `FaissIndexReorderTransformer` (remapped HNSW neighbor lists)
   e. Atomic rename of reordered files over originals

### Why finish() and not mergeOneField()

During `mergeOneField()`, the `.vec`/`.vemf` files are open `IndexOutput` streams without
finalized codec footers. `Lucene99FlatVectorsReader` validates the footer on open via
`CodecUtil.checkFooter()` and `CodecUtil.retrieveChecksum()`, which fails with:

```
CorruptIndexException: checksum status indeterminate: remaining=0
```

The footers are only written by `Lucene99FlatVectorsWriter.finish()`, which is called by
`KnnVectorsWriter.merge()` AFTER all `mergeOneField()` calls complete. So we must wait
until `finish()` to read the `.vec` file.

Additionally, `flatVectorsWriter.close()` must be called to flush the `IndexOutput` buffers
to disk before we can open the files with `Lucene99FlatVectorsReader`.

### Call chain during merge

```
SegmentMerger.mergeVectorValues()
  → try (KnnVectorsWriter writer = codec.fieldsWriter(state)) {
        writer.merge(mergeState);
     }
     // merge() internally:
     //   1. mergeOneField() per field  → writes .vec data, .faiss
     //   2. finish()                   → writes .vec/.vemf footers
     //                                 → closes flatVectorsWriter (flushes to disk)
     //                                 → reorders .vec + .vemf + .faiss  ← HERE
     // try-with-resources:
     //   3. close()                    → cleanup (flatVectorsWriter already closed)
```

### Files changed

- `NativeEngines990KnnVectorsWriter.java`: Added `fieldsToReorder` list populated in
  `mergeOneField()`, reorder loop in `finish()` after `flatVectorsWriter.finish()` + `close()`
- `KNN80CompoundFormat.java`: Removed all reorder logic, restored to original
- `SegmentReorderService.java`: Updated `reorderSegmentFiles()` to do full rewrite of
  `.vec` + `.vemf` + `.faiss` (was previously only rewriting `.faiss`)

### What about flush?

Flush is NOT reordered. The `fieldsToReorder` list is only populated by `mergeOneField()`,
not by `flush()`. Flush segments are typically small and will be merged later, at which point
reordering applies.
