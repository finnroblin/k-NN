/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.jni;

import com.google.common.collect.ImmutableMap;
import lombok.SneakyThrows;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.junit.BeforeClass;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.TestUtils;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.query.KNNQueryResult;
import org.opensearch.knn.index.store.IndexInputWithBuffer;

import java.io.IOException;
import java.net.URL;
import java.nio.file.Path;
import java.util.Collections;
import java.util.UUID;

import static org.opensearch.knn.common.KNNConstants.INDEX_DESCRIPTION_PARAMETER;

public class FaissServiceTests extends KNNTestCase {

    static TestUtils.TestData testData;
    static TestUtils.TestData testDataNested;
    static String faissBinaryMethod = "BHNSW32";

    @BeforeClass
    public static void setUpClass() throws IOException {
        if (FaissServiceTests.class.getClassLoader() == null) {
            throw new IllegalStateException("ClassLoader of FaissServiceTests Class is null");
        }
        URL testIndexVectors = JNIServiceTests.class.getClassLoader().getResource("data/test_vectors_1000x128.json");
        URL testIndexVectorsNested = JNIServiceTests.class.getClassLoader().getResource("data/test_vectors_nested_1000x128.json");
        URL testQueries = JNIServiceTests.class.getClassLoader().getResource("data/test_queries_100x128.csv");
        assert testIndexVectors != null;
        assert testIndexVectorsNested != null;
        assert testQueries != null;
        testData = new TestUtils.TestData(testIndexVectors.getPath(), testQueries.getPath());
        testDataNested = new TestUtils.TestData(testIndexVectorsNested.getPath(), testQueries.getPath());
    }

    @SneakyThrows
    public void testLoadIndexWithStreamADC() {
        Path tempDirPath = createTempDir();
        String indexFileName1 = "test1" + UUID.randomUUID() + ".tmp";
        try (Directory directory = newFSDirectory(tempDirPath)) {
            // Create an index with binary data
            long memoryAddr = testData.loadBinaryDataToMemoryAddress();
            TestUtils.createIndex(
                testData.indexData.docs,
                memoryAddr,
                testData.indexData.getDimension(),
                directory,
                indexFileName1,
                ImmutableMap.of(
                    INDEX_DESCRIPTION_PARAMETER,
                    faissBinaryMethod,
                    KNNConstants.SPACE_TYPE,
                    SpaceType.HAMMING.getValue(),
                    KNNConstants.VECTOR_DATA_TYPE_FIELD,
                    VectorDataType.BINARY.getValue()
                ),
                KNNEngine.FAISS
            );
            assertTrue(directory.fileLength(indexFileName1) > 0);

            try (IndexInput indexInput = directory.openInput(indexFileName1, IOContext.DEFAULT)) {
                long indexAddr = FaissService.loadIndexWithStreamADC(new IndexInputWithBuffer(indexInput));
                // Load the index for ADC, effectively converting it from a binary HNSW index to a regular HNSW index
                for (float[] query : testData.queries) {
                    KNNQueryResult[] results = JNIService.queryIndex(
                        indexAddr,
                        query,
                        10,
                        Collections.emptyMap(),
                        KNNEngine.FAISS,
                        null,
                        0,
                        null
                    );
                    // logQuery(results);
                    assertEquals(10, results.length);
                }
            }
        }
    }

    private void logQuery(KNNQueryResult[] results) {
        logger.info("Query results");
        for (KNNQueryResult result : results) {
            logger.info(result.getId() + ": " + result.getScore());
        }
    }
}
