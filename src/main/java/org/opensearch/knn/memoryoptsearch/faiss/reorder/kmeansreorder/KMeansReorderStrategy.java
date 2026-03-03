/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch.faiss.reorder.kmeansreorder;

import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.opensearch.knn.memoryoptsearch.faiss.reorder.VectorReorderStrategy;

import java.io.IOException;

/**
 * K-means clustering reorder strategy.
 * Note: KMeans requires float[][] in heap (JNI and pure-Java both need contiguous vector data).
 * Vectors are read from FloatVectorValues and materialized into a heap array.
 */
public class KMeansReorderStrategy implements VectorReorderStrategy {

    private static final int DEFAULT_K = 256;
    private static final int DEFAULT_NITER = 25;

    private final int k;
    private final int niter;

    public KMeansReorderStrategy() {
        this(DEFAULT_K, DEFAULT_NITER);
    }

    public KMeansReorderStrategy(int k, int niter) {
        this.k = k;
        this.niter = niter;
    }

    @Override
    public int[] computePermutation(FloatVectorValues vectors, int numThreads, VectorSimilarityFunction similarityFunction)
        throws IOException {
        int n = vectors.size();
        int effectiveK = Math.min(k, n);

        int metricType = (similarityFunction == VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT
            || similarityFunction == VectorSimilarityFunction.COSINE)
            ? KMeansClusterer.METRIC_INNER_PRODUCT
            : KMeansClusterer.METRIC_L2;

        // Materialize vectors into heap — required by KMeans clustering
        float[][] heapVectors = new float[n][];
        for (int i = 0; i < n; i++) {
            float[] src = vectors.vectorValue(i);
            heapVectors[i] = new float[src.length];
            System.arraycopy(src, 0, heapVectors[i], 0, src.length);
        }

        int[] perm = ClusterSorter.clusterAndSort(heapVectors, effectiveK, niter, metricType);
        return perm;
    }
}
