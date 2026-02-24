/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch.faiss.reorder;

import lombok.extern.log4j.Log4j2;
import org.apache.lucene.codecs.hnsw.DefaultFlatVectorScorer;
import org.apache.lucene.codecs.lucene99.Lucene99FlatVectorsReader;
import org.apache.lucene.index.DocValuesSkipIndexType;
import org.apache.lucene.index.DocValuesType;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FieldInfos;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.IndexOptions;
import org.apache.lucene.index.SegmentInfo;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.memoryoptsearch.faiss.FaissIdMapIndex;
import org.opensearch.knn.memoryoptsearch.faiss.FaissIndex;

import java.io.IOException;
import java.util.Collections;

import static org.opensearch.knn.index.codec.util.KNNCodecUtil.buildEngineFileName;

/**
 * Orchestrates post-write reordering of .vec and .faiss segment files.
 * Opens the .vec via Lucene99FlatVectorsReader (mmap-backed) and passes FloatVectorValues
 * to the VectorReorderStrategy — BP uses zero-heap-copy, KMeans materializes to heap.
 */
@Log4j2
public class SegmentReorderService {

    public static final int MIN_VECTORS_FOR_REORDER = 10_000;
    public static final int DEFAULT_REORDER_THREADS = 4;

    private static final String REORDER_SUFFIX = ".reorder";

    private final SegmentWriteState state;
    private final FieldInfo fieldInfo;
    private final VectorReorderStrategy strategy;
    private final int numThreads;

    public SegmentReorderService(SegmentWriteState state, FieldInfo fieldInfo, VectorReorderStrategy strategy) {
        this(state, fieldInfo, strategy, DEFAULT_REORDER_THREADS);
    }

    public SegmentReorderService(SegmentWriteState state, FieldInfo fieldInfo, VectorReorderStrategy strategy, int numThreads) {
        this.state = state;
        this.fieldInfo = fieldInfo;
        this.strategy = strategy;
        this.numThreads = numThreads;
    }

    /**
     * Reorder the .vec and .faiss files for the current segment/field.
     * 1. Open .vec via Lucene99FlatVectorsReader (mmap) to get FloatVectorValues
     * 2. Compute permutation via VectorReorderStrategy
     * 3. Rewrite .vec in reordered order
     * 4. Rewrite .faiss with remapped neighbor lists
     */
    public void reorderSegmentFiles() throws IOException {
        final Directory directory = state.directory;
        final String segmentName = state.segmentInfo.name;

        // Locate the .faiss file
        final KNNEngine knnEngine = KNNEngine.FAISS;
        final String engineFileName = buildEngineFileName(
            segmentName, knnEngine.getVersion(), fieldInfo.name, knnEngine.getExtension()
        );

        // Locate the .vec and .vemf files (Lucene99FlatVectorsFormat naming)
        final String vecDataFileName = findFileWithSuffix(directory, segmentName, ".vec");
        final String vecMetaFileName = findFileWithSuffix(directory, segmentName, ".vemf");

        if (vecDataFileName == null || vecMetaFileName == null) {
            log.warn("Cannot find .vec/.vemf files for segment {}, skipping reorder", segmentName);
            return;
        }

        // Step 1: Open .vec via mmap-backed reader and compute permutation
        final int[] permutation;
        try {
            permutation = computePermutationFromVecFile(directory, segmentName, vecDataFileName, vecMetaFileName);
        } catch (Exception e) {
            log.error("Failed to compute reorder permutation for segment {}, field {}", segmentName, fieldInfo.name, e);
            return;
        }

        final ReorderOrdMap reorderOrdMap = new ReorderOrdMap(permutation);

        // Step 2: Rewrite .vec file in reordered order
        rewriteVecFile(directory, vecDataFileName, vecMetaFileName, reorderOrdMap);

        // Step 3: Rewrite .faiss file with remapped neighbor lists
        rewriteFaissFile(directory, engineFileName, reorderOrdMap);
    }

    private int[] computePermutationFromVecFile(
        Directory directory, String segmentName, String vecDataFileName, String vecMetaFileName
    ) throws IOException {
        // Build a SegmentReadState to open the Lucene99FlatVectorsReader
        final FieldInfo readFieldInfo = buildFieldInfoForRead();
        final FieldInfos fieldInfos = new FieldInfos(new FieldInfo[] { readFieldInfo });

        // We need the segment suffix that Lucene99FlatVectorsFormat uses.
        // Extract it from the .vec filename: format is {segmentName}_{suffix}_{fieldName}.vec
        final String segmentSuffix = extractSegmentSuffix(vecDataFileName, segmentName);

        final SegmentReadState readState = new SegmentReadState(
            directory, state.segmentInfo, fieldInfos, IOContext.DEFAULT, segmentSuffix
        );

        try (Lucene99FlatVectorsReader reader = new Lucene99FlatVectorsReader(readState, DefaultFlatVectorScorer.INSTANCE)) {
            final FloatVectorValues vectorValues = reader.getFloatVectorValues(fieldInfo.name);
            if (vectorValues == null) {
                throw new IOException("No float vector values for field " + fieldInfo.name);
            }
            log.info("Computing reorder permutation for {} vectors, field [{}]", vectorValues.size(), fieldInfo.name);
            return strategy.computePermutation(vectorValues, numThreads);
        }
    }

    private void rewriteVecFile(
        Directory directory, String vecDataFileName, String vecMetaFileName, ReorderOrdMap reorderOrdMap
    ) throws IOException {
        final String reorderedVecData = vecDataFileName + REORDER_SUFFIX;
        final String reorderedVecMeta = vecMetaFileName + REORDER_SUFFIX;

        // Clean up any leftover reorder files
        deleteIfExists(directory, reorderedVecData);
        deleteIfExists(directory, reorderedVecMeta);

        final int numVectors = reorderOrdMap.newOrd2Old.length;

        // Build reader state
        final FieldInfo readFieldInfo = buildFieldInfoForRead();
        final FieldInfos fieldInfos = new FieldInfos(new FieldInfo[] { readFieldInfo });
        final String segmentSuffix = extractSegmentSuffix(vecDataFileName, state.segmentInfo.name);

        final SegmentReadState readState = new SegmentReadState(
            directory, state.segmentInfo, fieldInfos, IOContext.DEFAULT, segmentSuffix
        );

        try (IndexInput vecMetaInput = directory.openInput(vecMetaFileName, IOContext.DEFAULT)) {
            final ReorderedFlatVectorsWriter writer = new ReorderedFlatVectorsWriter(
                directory, reorderedVecMeta, reorderedVecData, numVectors, vecMetaInput
            );

            try (
                writer;
                Lucene99FlatVectorsReader reader = new Lucene99FlatVectorsReader(readState, DefaultFlatVectorScorer.INSTANCE)
            ) {
                final FloatVectorValues vectorValues = reader.getFloatVectorValues(fieldInfo.name);
                final ReorderedFlatVectorsWriter.ReorderedFlatFieldVectorsWriter fieldWriter = writer.addField(readFieldInfo);

                for (int newOrd = 0; newOrd < numVectors; newOrd++) {
                    int oldOrd = reorderOrdMap.newOrd2Old[newOrd];
                    float[] vector = vectorValues.vectorValue(oldOrd);
                    fieldWriter.addValue(oldOrd, vector);
                }
                fieldWriter.finish();
            }
        }

        // Atomic replace: delete originals, rename reordered files
        directory.deleteFile(vecDataFileName);
        directory.deleteFile(vecMetaFileName);
        directory.rename(reorderedVecData, vecDataFileName);
        directory.rename(reorderedVecMeta, vecMetaFileName);

        log.info("Reordered .vec file for segment {}, field [{}]", state.segmentInfo.name, fieldInfo.name);
    }

    private void rewriteFaissFile(Directory directory, String engineFileName, ReorderOrdMap reorderOrdMap) throws IOException {
        final String reorderedFaiss = engineFileName + REORDER_SUFFIX;
        deleteIfExists(directory, reorderedFaiss);

        try (IndexInput faissInput = directory.openInput(engineFileName, IOContext.DEFAULT)) {
            final FaissIndex faissIndex = FaissIndex.load(faissInput);

            try (IndexOutput faissOutput = directory.createOutput(reorderedFaiss, IOContext.DEFAULT)) {
                FaissIndexReorderTransformer.transform(faissIndex, faissInput, faissOutput, reorderOrdMap);
            }
        }

        directory.deleteFile(engineFileName);
        directory.rename(reorderedFaiss, engineFileName);

        log.info("Reordered .faiss file for segment {}, field [{}]", state.segmentInfo.name, fieldInfo.name);
    }

    private FieldInfo buildFieldInfoForRead() {
        return new FieldInfo(
            fieldInfo.name,
            fieldInfo.number,
            false, false, false,
            IndexOptions.NONE,
            DocValuesType.NONE,
            DocValuesSkipIndexType.NONE,
            -1,
            Collections.emptyMap(),
            0, 0, 0,
            fieldInfo.getVectorDimension(),
            VectorEncoding.FLOAT32,
            fieldInfo.getVectorSimilarityFunction(),
            false, false
        );
    }

    /**
     * Extract the segment suffix from a .vec filename.
     * Filename format: {segmentName}_{suffix}_{fieldName}.vec
     * e.g. "_0_NativeEngines990KnnVectorsFormat_0.vec" → "NativeEngines990KnnVectorsFormat_0"
     */
    private String extractSegmentSuffix(String vecFileName, String segmentName) {
        // Strip segment prefix (e.g. "_0_") and the ".vec" extension
        String withoutPrefix = vecFileName.substring(segmentName.length() + 1); // skip "{segmentName}_"
        // The suffix is everything between the segment prefix and the field-specific part
        // Lucene99FlatVectorsFormat uses: segmentName + "_" + segmentSuffix + "_" + ...
        // But the actual suffix used in SegmentReadState is what was passed to the format.
        // For NativeEngines990KnnVectorsFormat, the suffix comes from SegmentWriteState.
        return state.segmentSuffix;
    }

    private String findFileWithSuffix(Directory directory, String segmentName, String extension) throws IOException {
        for (String file : directory.listAll()) {
            if (file.startsWith(segmentName + "_") && file.endsWith(extension)) {
                return file;
            }
        }
        return null;
    }

    private static void deleteIfExists(Directory directory, String fileName) {
        try {
            directory.deleteFile(fileName);
        } catch (IOException ignored) {}
    }
}
