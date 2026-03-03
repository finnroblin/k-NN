/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch.faiss.reorder;

import org.apache.lucene.index.VectorSimilarityFunction;
import org.opensearch.knn.memoryoptsearch.faiss.reorder.bpreorder.BpReorderer;

/**
 * Bipartite graph partitioning reorder strategy.
 * Delegates to Lucene's BpVectorReorderer for optimal cache-line locality.
 */
public class BipartiteReorderStrategy implements VectorReorderStrategy {

    @Override
    public int[] computePermutation(float[][] vectors, int numThreads) {
        return BpReorderer.computePermutation(vectors, VectorSimilarityFunction.EUCLIDEAN);
    }
}
