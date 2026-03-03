/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch.faiss.reorder;

import org.opensearch.knn.memoryoptsearch.faiss.reorder.kmeansreorder.ClusterSorter;
import org.opensearch.knn.memoryoptsearch.faiss.reorder.kmeansreorder.KMeansClusterer;

/**
 * K-Means clustering reorder strategy.
 * Groups vectors by cluster assignment then sorts within each cluster by distance to centroid.
 */
public class KMeansReorderStrategy implements VectorReorderStrategy {

    private static final int DEFAULT_NUM_ITERATIONS = 25;

    private final int numClusters;

    public KMeansReorderStrategy(int numClusters) {
        this.numClusters = numClusters;
    }

    @Override
    public int[] computePermutation(float[][] vectors, int numThreads) {
        return ClusterSorter.clusterAndSort(vectors, numClusters, DEFAULT_NUM_ITERATIONS, KMeansClusterer.METRIC_L2);
    }
}
