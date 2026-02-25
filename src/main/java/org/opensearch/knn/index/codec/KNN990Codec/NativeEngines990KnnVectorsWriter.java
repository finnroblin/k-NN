/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 *
 * Modifications Copyright OpenSearch Contributors. See
 * GitHub history for details.
 */

package org.opensearch.knn.index.codec.KNN990Codec;

import lombok.extern.log4j.Log4j2;
import org.apache.lucene.codecs.KnnFieldVectorsWriter;
import org.apache.lucene.codecs.KnnVectorsWriter;
import org.apache.lucene.codecs.hnsw.FlatVectorsWriter;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.MergeState;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.index.Sorter;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.util.IOUtils;
import org.apache.lucene.util.RamUsageEstimator;
import org.opensearch.common.StopWatch;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.codec.nativeindex.NativeIndexBuildStrategyFactory;
import org.opensearch.knn.index.codec.nativeindex.NativeIndexWriter;
import org.opensearch.knn.index.quantizationservice.QuantizationService;
import org.opensearch.knn.index.vectorvalues.KNNVectorValues;
import org.opensearch.knn.memoryoptsearch.faiss.reorder.SegmentReorderService;
import org.opensearch.knn.memoryoptsearch.faiss.reorder.VectorReorderStrategy;
import org.opensearch.knn.plugin.stats.KNNGraphValue;
import org.opensearch.knn.quantization.models.quantizationParams.QuantizationParams;
import org.opensearch.knn.quantization.models.quantizationState.QuantizationState;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Supplier;

import static org.opensearch.knn.common.FieldInfoExtractor.extractVectorDataType;
import static org.opensearch.knn.index.vectorvalues.KNNVectorValuesFactory.getKNNVectorValuesSupplierForMerge;
import static org.opensearch.knn.index.vectorvalues.KNNVectorValuesFactory.getVectorValuesSupplier;

/**
 * A KNNVectorsWriter class for writing the vector data structures and flat vectors for Native Engines.
 */
@Log4j2
public class NativeEngines990KnnVectorsWriter extends KnnVectorsWriter {
    private static final long SHALLOW_SIZE = RamUsageEstimator.shallowSizeOfInstance(NativeEngines990KnnVectorsWriter.class);

    private final SegmentWriteState segmentWriteState;
    private final FlatVectorsWriter flatVectorsWriter;
    private KNN990QuantizationStateWriter quantizationStateWriter;
    private final List<NativeEngineFieldVectorsWriter<?>> fields = new ArrayList<>();
    private boolean finished;
    private final Integer approximateThreshold;
    private final NativeIndexBuildStrategyFactory nativeIndexBuildStrategyFactory;
    private final VectorReorderStrategy reorderStrategy;

    // Fields that need reordering after finish() writes footers.
    // Populated during mergeOneField() for fields above the reorder threshold.
    private final List<FieldInfo> fieldsToReorder = new ArrayList<>();

    public NativeEngines990KnnVectorsWriter(
        SegmentWriteState segmentWriteState,
        FlatVectorsWriter flatVectorsWriter,
        Integer approximateThreshold,
        NativeIndexBuildStrategyFactory nativeIndexBuildStrategyFactory
    ) {
        this(segmentWriteState, flatVectorsWriter, approximateThreshold, nativeIndexBuildStrategyFactory, null);
    }

    public NativeEngines990KnnVectorsWriter(
        SegmentWriteState segmentWriteState,
        FlatVectorsWriter flatVectorsWriter,
        Integer approximateThreshold,
        NativeIndexBuildStrategyFactory nativeIndexBuildStrategyFactory,
        VectorReorderStrategy reorderStrategy
    ) {
        this.segmentWriteState = segmentWriteState;
        this.flatVectorsWriter = flatVectorsWriter;
        this.approximateThreshold = approximateThreshold;
        this.nativeIndexBuildStrategyFactory = nativeIndexBuildStrategyFactory;
        this.reorderStrategy = reorderStrategy;
    }

    @Override
    public KnnFieldVectorsWriter<?> addField(final FieldInfo fieldInfo) throws IOException {
        final NativeEngineFieldVectorsWriter<?> newField = NativeEngineFieldVectorsWriter.create(
            fieldInfo,
            flatVectorsWriter.addField(fieldInfo),
            segmentWriteState.infoStream
        );
        fields.add(newField);
        return newField;
    }

    @Override
    public void flush(int maxDoc, final Sorter.DocMap sortMap) throws IOException {
        flatVectorsWriter.flush(maxDoc, sortMap);

        for (final NativeEngineFieldVectorsWriter<?> field : fields) {
            final FieldInfo fieldInfo = field.getFieldInfo();
            final VectorDataType vectorDataType = extractVectorDataType(fieldInfo);
            int totalLiveDocs = field.getVectors().size();
            if (totalLiveDocs == 0) {
                log.debug("[Flush] No live docs for field {}", fieldInfo.getName());
                continue;
            }
            final Supplier<KNNVectorValues<?>> knnVectorValuesSupplier = getVectorValuesSupplier(
                vectorDataType,
                field.getFlatFieldVectorsWriter().getDocsWithFieldSet(),
                field.getVectors()
            );
            final QuantizationState quantizationState = train(field.getFieldInfo(), knnVectorValuesSupplier, totalLiveDocs);
            if (quantizationState == null && shouldSkipBuildingVectorDataStructure(totalLiveDocs)) {
                log.debug(
                    "Skip building vector data structure for field: {}, as liveDoc: {} is less than the threshold {} during flush",
                    fieldInfo.name,
                    totalLiveDocs,
                    approximateThreshold
                );
                continue;
            }
            final NativeIndexWriter writer = NativeIndexWriter.getWriter(
                fieldInfo,
                segmentWriteState,
                quantizationState,
                nativeIndexBuildStrategyFactory
            );

            StopWatch stopWatch = new StopWatch().start();
            writer.flushIndex(knnVectorValuesSupplier, totalLiveDocs);
            long time_in_millis = stopWatch.stop().totalTime().millis();
            KNNGraphValue.REFRESH_TOTAL_TIME_IN_MILLIS.incrementBy(time_in_millis);
            log.debug("Flush took {} ms for vector field [{}]", time_in_millis, fieldInfo.getName());
        }
    }

    @Override
    public void mergeOneField(final FieldInfo fieldInfo, final MergeState mergeState) throws IOException {
        flatVectorsWriter.mergeOneField(fieldInfo, mergeState);

        final VectorDataType vectorDataType = extractVectorDataType(fieldInfo);
        final Supplier<KNNVectorValues<?>> knnVectorValuesSupplier = getKNNVectorValuesSupplierForMerge(
            vectorDataType,
            fieldInfo,
            mergeState
        );
        int totalLiveDocs = getLiveDocs(knnVectorValuesSupplier.get());
        if (totalLiveDocs == 0) {
            log.debug("[Merge] No live docs for field {}", fieldInfo.getName());
            return;
        }

        final QuantizationState quantizationState = train(fieldInfo, knnVectorValuesSupplier, totalLiveDocs);
        if (quantizationState == null && shouldSkipBuildingVectorDataStructure(totalLiveDocs)) {
            log.debug(
                "Skip building vector data structure for field: {}, as liveDoc: {} is less than the threshold {} during merge",
                fieldInfo.name,
                totalLiveDocs,
                approximateThreshold
            );
            return;
        }
        final NativeIndexWriter writer = NativeIndexWriter.getWriter(
            fieldInfo,
            segmentWriteState,
            quantizationState,
            nativeIndexBuildStrategyFactory
        );

        StopWatch stopWatch = new StopWatch().start();
        writer.mergeIndex(knnVectorValuesSupplier, totalLiveDocs);
        long time_in_millis = stopWatch.stop().totalTime().millis();
        KNNGraphValue.MERGE_TOTAL_TIME_IN_MILLIS.incrementBy(time_in_millis);
        log.debug("Merge took {} ms for vector field [{}]", time_in_millis, fieldInfo.getName());

        // Mark this field for reordering. The actual reorder happens in finish() after
        // flatVectorsWriter.finish() writes the .vec/.vemf footers, so we can read them.
        if (reorderStrategy != null && totalLiveDocs >= SegmentReorderService.MIN_VECTORS_FOR_REORDER) {
            fieldsToReorder.add(fieldInfo);
        }
    }

    @Override
    public void finish() throws IOException {
        if (finished) {
            throw new IllegalStateException("NativeEnginesKNNVectorsWriter is already finished");
        }
        finished = true;
        if (quantizationStateWriter != null) {
            quantizationStateWriter.writeFooter();
        }
        // This writes the .vec/.vemf codec footers.
        flatVectorsWriter.finish();
        // Close to flush the IndexOutput buffers to disk so we can read the files.
        flatVectorsWriter.close();

        // Now that .vec/.vemf are finalized and flushed to disk, reorder any fields
        // that were marked during mergeOneField(). Rewrites .vec, .vemf, and .faiss.
        for (FieldInfo fieldInfo : fieldsToReorder) {
            try {
                StopWatch reorderWatch = new StopWatch().start();
                SegmentReorderService reorderService = new SegmentReorderService(
                    segmentWriteState, fieldInfo, reorderStrategy
                );
                reorderService.reorderSegmentFiles();
                long reorderMs = reorderWatch.stop().totalTime().millis();
                log.info("Reorder took {} ms for field [{}]", reorderMs, fieldInfo.getName());
            } catch (Exception e) {
                log.error("Failed to reorder field [{}], continuing without reorder", fieldInfo.getName(), e);
            }
        }
    }

    @Override
    public void close() throws IOException {
        if (quantizationStateWriter != null) {
            quantizationStateWriter.closeOutput();
        }
        IOUtils.close(flatVectorsWriter);
    }

    @Override
    public long ramBytesUsed() {
        return SHALLOW_SIZE + flatVectorsWriter.ramBytesUsed() + fields.stream()
            .mapToLong(NativeEngineFieldVectorsWriter::ramBytesUsed)
            .sum();
    }

    private QuantizationState train(
        final FieldInfo fieldInfo,
        final Supplier<KNNVectorValues<?>> knnVectorValuesSupplier,
        final int totalLiveDocs
    ) throws IOException {
        final QuantizationService quantizationService = QuantizationService.getInstance();
        final QuantizationParams quantizationParams = quantizationService.getQuantizationParams(fieldInfo);
        QuantizationState quantizationState = null;
        if (quantizationParams != null && totalLiveDocs > 0) {
            initQuantizationStateWriterIfNecessary();
            quantizationState = quantizationService.train(quantizationParams, knnVectorValuesSupplier, totalLiveDocs);
            quantizationStateWriter.writeState(fieldInfo.getFieldNumber(), quantizationState);
        }
        return quantizationState;
    }

    private int getLiveDocs(KNNVectorValues<?> vectorValues) throws IOException {
        int liveDocs = 0;
        while (vectorValues.nextDoc() != DocIdSetIterator.NO_MORE_DOCS) {
            liveDocs++;
        }
        return liveDocs;
    }

    private void initQuantizationStateWriterIfNecessary() throws IOException {
        if (quantizationStateWriter == null) {
            quantizationStateWriter = new KNN990QuantizationStateWriter(segmentWriteState);
            quantizationStateWriter.writeHeader(segmentWriteState);
        }
    }

    private boolean shouldSkipBuildingVectorDataStructure(final long docCount) {
        if (approximateThreshold < 0) {
            return true;
        }
        return docCount < approximateThreshold;
    }
}
