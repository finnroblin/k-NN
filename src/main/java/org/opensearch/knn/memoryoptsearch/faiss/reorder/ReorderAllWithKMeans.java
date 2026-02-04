/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch.faiss.reorder;

import org.apache.lucene.codecs.hnsw.DefaultFlatVectorScorer;
import org.apache.lucene.codecs.lucene99.Lucene99FlatVectorsReader;
import org.apache.lucene.index.DocValuesSkipIndexType;
import org.apache.lucene.index.DocValuesType;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FieldInfos;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.IndexOptions;
import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.index.SegmentInfo;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.store.MMapDirectory;
import org.opensearch.knn.memoryoptsearch.faiss.FaissIdMapIndex;
import org.opensearch.knn.memoryoptsearch.faiss.FaissIndex;
import org.opensearch.knn.memoryoptsearch.faiss.reorder.kmeansreorder.ClusterSorter;
import org.opensearch.knn.memoryoptsearch.faiss.reorder.kmeansreorder.KMeansClusterer;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.NoSuchFileException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import static org.apache.lucene.search.DocIdSetIterator.NO_MORE_DOCS;

public class ReorderAllWithKMeans {
    private static final String reorderSuffix = ".reorder";
    private static final int DEFAULT_NUM_CLUSTERS = 500;
    private static final int DEFAULT_NUM_ITERATIONS = 25;

    public record TargetFiles(String faissIndexFileName, String flatVectorDataFileName, String flatVectorMetaFileName,
        String engineLuceneDirectory) {
    }

    public static void main(String... args) throws IOException {
        final String searchDirectory = "/home/ec2-user/k-NN/build/testclusters/integTest-0/distro/3.5.0-ARCHIVE/data";

        final List<TargetFiles> targetFilesList = findTargetFiles(Path.of(searchDirectory));
        for (final TargetFiles targetFiles : targetFilesList) {
            System.out.println();
            System.out.println("Start K-Means reordering ...");
            System.out.println("  Lucene dir: " + targetFiles.engineLuceneDirectory);
            System.out.println("  Vec: " + targetFiles.flatVectorDataFileName);
            System.out.println("  VecMeta: " + targetFiles.flatVectorMetaFileName);
            System.out.println("  Faiss: " + targetFiles.faissIndexFileName);
            reorder(targetFiles);
        }
    }

    public static List<TargetFiles> findTargetFiles(Path rootDir) throws IOException {
        Objects.requireNonNull(rootDir, "rootDir must not be null");

        Map<Path, List<String>> filesByDir;
        try (Stream<Path> stream = Files.walk(rootDir)) {
            filesByDir = stream.filter(Files::isRegularFile)
                .collect(Collectors.groupingBy(Path::getParent, Collectors.mapping(p -> p.getFileName().toString(), Collectors.toList())));
        }

        final List<TargetFiles> results = new ArrayList<>();

        for (Map.Entry<Path, List<String>> entry : filesByDir.entrySet()) {
            Path dir = entry.getKey();
            List<String> fileNames = entry.getValue();

            Map<String, List<String>> filesBySegment = new java.util.HashMap<>();
            for (String fileName : fileNames) {
                if (fileName.endsWith(".faiss") || fileName.endsWith(".vec") || fileName.endsWith(".vemf")) {
                    int secondUnderscore = fileName.indexOf('_', 1);
                    if (secondUnderscore > 0) {
                        String segmentPrefix = fileName.substring(0, secondUnderscore);
                        filesBySegment.computeIfAbsent(segmentPrefix, k -> new ArrayList<>()).add(fileName);
                    }
                }
            }

            for (Map.Entry<String, List<String>> segmentEntry : filesBySegment.entrySet()) {
                List<String> segmentFiles = segmentEntry.getValue();
                String faiss = findSingle(segmentFiles, ".faiss", dir);
                String vec = findSingle(segmentFiles, ".vec", dir);
                String vemf = findSingle(segmentFiles, ".vemf", dir);

                if (faiss != null && vec != null && vemf != null) {
                    results.add(new TargetFiles(faiss, vec, vemf, dir.toAbsolutePath().toString()));
                }
            }
        }

        return results;
    }

    private static String findSingle(List<String> names, String extension, Path dir) {
        List<String> matches = names.stream().filter(n -> n.endsWith(extension)).toList();
        if (matches.size() > 1) {
            throw new IllegalStateException("Multiple " + extension + " files found in: " + dir);
        }
        return matches.isEmpty() ? null : matches.get(0);
    }

    private static void reorder(TargetFiles targetFiles) throws IOException {
        final String segmentName = targetFiles.flatVectorDataFileName.substring(0, targetFiles.flatVectorDataFileName.indexOf('_', 1));
        final String fieldName = "target_field";
        final int fieldNo = 5;

        try (final Directory directory = new MMapDirectory(Path.of(targetFiles.engineLuceneDirectory))) {
            try (final IndexInput faissIndexInput = directory.openInput(targetFiles.faissIndexFileName, IOContext.DEFAULT)) {
                final FaissIndex faissIndex = FaissIndex.load(faissIndexInput);
                if (faissIndex instanceof FaissIdMapIndex idMapIndex) {
                    final int numVectors = faissIndex.getTotalNumberOfVectors();
                    final int dimension = faissIndex.getDimension();
                    
                    // Get similarity function, with fallback for binary indices
                    VectorSimilarityFunction similarityFunction;
                    try {
                        similarityFunction = faissIndex.getVectorSimilarityFunction().getVectorSimilarityFunction();
                    } catch (IllegalStateException e) {
                        // Fallback for indices where VectorSimilarityFunction is not available
                        similarityFunction = VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT;
                    }
                    
                    System.out.println("Total #vectors: " + numVectors + " (extracted from faiss index)");
                    System.out.println("Dimension: " + dimension + " (extracted from faiss index)");
                    System.out.println("Similarity: " + similarityFunction);

                    long s = System.nanoTime();
                    final ReorderOrdMap reorderOrdMap = getOrderMap(idMapIndex, faissIndexInput, targetFiles, directory, fieldName, dimension, fieldNo, segmentName, similarityFunction);
                    long e = System.nanoTime();
                    System.out.println("K-Means Reordering took : " + (e - s) / 1e6 + "ms");

                    for (final String reorderedFile : Arrays.asList(
                        targetFiles.flatVectorDataFileName + reorderSuffix,
                        targetFiles.flatVectorMetaFileName + reorderSuffix
                    )) {
                        try {
                            directory.deleteFile(reorderedFile);
                        } catch (NoSuchFileException x) {}
                    }

                    System.out.println("Transforming .vec file: " + targetFiles.flatVectorDataFileName + " ...");
                    s = System.nanoTime();
                    final ReorderedFlatVectorsWriter flatVectorsWriter;
                    try (final IndexInput vecMetaInput = directory.openInput(targetFiles.flatVectorMetaFileName, IOContext.DEFAULT)) {
                        flatVectorsWriter = new ReorderedFlatVectorsWriter(
                            directory,
                            targetFiles.flatVectorMetaFileName + reorderSuffix,
                            targetFiles.flatVectorDataFileName + reorderSuffix,
                            faissIndex.getTotalNumberOfVectors(),
                            vecMetaInput
                        );
                    }

                    final SegmentInfo segmentInfo = new SegmentInfo(
                        directory,
                        org.apache.lucene.util.Version.LATEST,
                        org.apache.lucene.util.Version.LATEST,
                        segmentName,
                        numVectors,
                        false,
                        false,
                        null,
                        Collections.emptyMap(),
                        flatVectorsWriter.segmentId,
                        Collections.emptyMap(),
                        null
                    );

                    final FieldInfo[] fieldInfoArr = new FieldInfo[6];
                    for (int i = 0; i < fieldInfoArr.length; ++i) {
                        final String targetFieldName = i != 5 ? "dummy-" + i : fieldName;
                        fieldInfoArr[i] = new FieldInfo(
                            targetFieldName,
                            i,
                            false,
                            false,
                            false,
                            IndexOptions.NONE,
                            DocValuesType.NONE,
                            DocValuesSkipIndexType.NONE,
                            -1,
                            Collections.emptyMap(),
                            0,
                            0,
                            0,
                            dimension,
                            VectorEncoding.FLOAT32,
                            similarityFunction,
                            false,
                            false
                        );
                    }
                    final FieldInfos fieldInfos = new FieldInfos(fieldInfoArr);

                    final SegmentReadState readState = new SegmentReadState(
                        directory,
                        segmentInfo,
                        fieldInfos,
                        IOContext.DEFAULT,
                        flatVectorsWriter.segmentSuffix
                    );

                    try (
                        flatVectorsWriter;
                        Lucene99FlatVectorsReader flatVectorsReader = new Lucene99FlatVectorsReader(
                            readState,
                            DefaultFlatVectorScorer.INSTANCE
                        )
                    ) {
                        final FloatVectorValues vectorValues = flatVectorsReader.getFloatVectorValues(fieldName);

                        final ReorderedFlatVectorsWriter.ReorderedFlatFieldVectorsWriter writer = flatVectorsWriter.addField(
                            fieldInfoArr[fieldNo]
                        );
                        for (int ord = 0; ord < reorderOrdMap.newOrd2Old.length; ++ord) {
                            final float[] vector = vectorValues.vectorValue(reorderOrdMap.newOrd2Old[ord]);
                            writer.addValue(reorderOrdMap.newOrd2Old[ord], vector);
                        }
                        writer.finish();
                    }
                    e = System.nanoTime();
                    System.out.println("Transforming .vec took " + ((e - s) / 1e6 + "ms"));
                } else {
                    throw new IllegalStateException("faissIndex is not FaissIdMapIndex! Actual type: " + faissIndex.getClass());
                }
            }
        }

        System.out.println("Reorder is done!");
        System.out.println("Overriding files ...");
        ReorderAll.switchFiles(Path.of(targetFiles.engineLuceneDirectory), targetFiles.faissIndexFileName, reorderSuffix);
        ReorderAll.switchFiles(Path.of(targetFiles.engineLuceneDirectory), targetFiles.flatVectorDataFileName, reorderSuffix);
        ReorderAll.switchFiles(Path.of(targetFiles.engineLuceneDirectory), targetFiles.flatVectorMetaFileName, reorderSuffix);
    }

    private static ReorderOrdMap getOrderMap(
        FaissIdMapIndex idMapIndex,
        IndexInput faissIndexInput,
        TargetFiles targetFiles,
        Directory directory,
        String fieldName,
        int dimension,
        int fieldNo,
        String segmentName,
        VectorSimilarityFunction similarityFunction
    ) throws IOException {
        final int numVectors = idMapIndex.getTotalNumberOfVectors();

        System.out.println("Loading vectors from .vec file for K-Means reordering...");
        float[][] vectors = loadVectorsFromVec(directory, targetFiles, fieldName, dimension, fieldNo, segmentName, numVectors, similarityFunction);

        // Determine number of clusters (k)
        int k = Math.min(DEFAULT_NUM_CLUSTERS, numVectors / 100);
        k = Math.max(k, 2); // At least 2 clusters
        System.out.println("Computing K-Means permutation with k=" + k + " clusters...");

        // Use inner product metric for MAXIMUM_INNER_PRODUCT similarity
        int metricType = (similarityFunction == VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT) 
            ? KMeansClusterer.METRIC_INNER_PRODUCT 
            : KMeansClusterer.METRIC_L2;
        int[] permutation = ClusterSorter.clusterAndSort(vectors, k, DEFAULT_NUM_ITERATIONS, metricType);

        ReorderOrdMap reorderOrdMap = new ReorderOrdMap(permutation);
        // Transform the faiss index
        try (final IndexOutput indexOutput = directory.createOutput(targetFiles.faissIndexFileName + reorderSuffix, IOContext.DEFAULT)) {
            FaissIndexReorderTransformer.transform(idMapIndex, faissIndexInput, indexOutput, reorderOrdMap);
        }

        return reorderOrdMap;
    }

    private static float[][] loadVectorsFromVec(
        Directory directory,
        TargetFiles targetFiles,
        String fieldName,
        int dimension,
        int fieldNo,
        String segmentName,
        int numVectors,
        VectorSimilarityFunction similarityFunction
    ) throws IOException {
        try {
            directory.deleteFile(targetFiles.flatVectorMetaFileName + ".tmp");
        } catch (NoSuchFileException ignored) {}
        try {
            directory.deleteFile(targetFiles.flatVectorDataFileName + ".tmp");
        } catch (NoSuchFileException ignored) {}

        final ReorderedFlatVectorsWriter tempWriter;
        try (final IndexInput vecMetaInput = directory.openInput(targetFiles.flatVectorMetaFileName, IOContext.DEFAULT)) {
            tempWriter = new ReorderedFlatVectorsWriter(
                directory,
                targetFiles.flatVectorMetaFileName + ".tmp",
                targetFiles.flatVectorDataFileName + ".tmp",
                numVectors,
                vecMetaInput
            );
        }

        final SegmentInfo segmentInfo = new SegmentInfo(
            directory,
            org.apache.lucene.util.Version.LATEST,
            org.apache.lucene.util.Version.LATEST,
            segmentName,
            numVectors,
            false,
            false,
            null,
            Collections.emptyMap(),
            tempWriter.segmentId,
            Collections.emptyMap(),
            null
        );

        final FieldInfo[] fieldInfoArr = new FieldInfo[6];
        for (int i = 0; i < fieldInfoArr.length; ++i) {
            final String targetFieldName = i != 5 ? "dummy-" + i : fieldName;
            fieldInfoArr[i] = new FieldInfo(
                targetFieldName,
                i,
                false,
                false,
                false,
                IndexOptions.NONE,
                DocValuesType.NONE,
                DocValuesSkipIndexType.NONE,
                -1,
                Collections.emptyMap(),
                0,
                0,
                0,
                dimension,
                VectorEncoding.FLOAT32,
                similarityFunction,
                false,
                false
            );
        }
        final FieldInfos fieldInfos = new FieldInfos(fieldInfoArr);

        final SegmentReadState readState = new SegmentReadState(
            directory,
            segmentInfo,
            fieldInfos,
            IOContext.DEFAULT,
            tempWriter.segmentSuffix
        );

        float[][] vectors = new float[numVectors][];
        try (
            tempWriter;
            Lucene99FlatVectorsReader flatVectorsReader = new Lucene99FlatVectorsReader(
                readState,
                DefaultFlatVectorScorer.INSTANCE
            )
        ) {
            final FloatVectorValues vectorValues = flatVectorsReader.getFloatVectorValues(fieldName);
            for (int ord = 0; ord < numVectors; ++ord) {
                vectors[ord] = vectorValues.vectorValue(ord).clone();
            }
        }

        try {
            directory.deleteFile(targetFiles.flatVectorMetaFileName + ".tmp");
        } catch (NoSuchFileException ignored) {}
        try {
            directory.deleteFile(targetFiles.flatVectorDataFileName + ".tmp");
        } catch (NoSuchFileException ignored) {}

        return vectors;
    }
}
