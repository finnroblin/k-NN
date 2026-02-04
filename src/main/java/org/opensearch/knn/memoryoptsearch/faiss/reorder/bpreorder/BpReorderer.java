/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch.faiss.reorder.bpreorder;

import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.Sorter;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.misc.index.BpVectorReorderer;

import java.util.Arrays;

/**
 * Bipartite graph partitioning reorderer for vectors.
 * Uses Lucene's BpVectorReorderer to compute optimal ordering.
 */
public class BpReorderer {

    private static final String DUMMY_FIELD = "vectors";

    /**
     * Compute reordering permutation using BP algorithm.
     * @param vectors input vectors
     * @return newOrder where newOrder[newIdx] = oldIdx
     */
    public static int[] computePermutation(float[][] vectors) {
        return computePermutation(vectors, VectorSimilarityFunction.EUCLIDEAN);
    }

    /**
     * Compute reordering permutation using BP algorithm with specified similarity.
     * @param vectors input vectors
     * @param similarity vector similarity function
     * @return newOrder where newOrder[newIdx] = oldIdx
     */
    public static int[] computePermutation(float[][] vectors, VectorSimilarityFunction similarity) {
        int n = vectors.length;
        int dim = vectors[0].length;

        BpVectorReorderer reorderer = new BpVectorReorderer(DUMMY_FIELD);
        reorderer.setMinPartitionSize(1);

        FloatVectorValues fvv = FloatVectorValues.fromFloats(Arrays.asList(vectors), dim); // TODO: repeated loads
        Sorter.DocMap map = reorderer.computeValueMap(fvv, similarity, null);

        int[] newOrder = new int[n];
        for (int i = 0; i < n; i++) {
            newOrder[i] = map.newToOld(i);
        }
        return newOrder;
    }
}
