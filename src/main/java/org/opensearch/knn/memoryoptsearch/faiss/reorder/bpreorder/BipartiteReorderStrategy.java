/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch.faiss.reorder.bpreorder;

import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.Sorter;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.misc.index.BpVectorReorderer;
import org.apache.lucene.search.TaskExecutor;
import org.opensearch.knn.memoryoptsearch.faiss.reorder.VectorReorderStrategy;

import java.io.IOException;
import java.util.concurrent.ForkJoinPool;

/**
 * Bipartite graph partitioning reorder strategy.
 * Passes FloatVectorValues directly to BpVectorReorderer — supports mmap-backed vectors
 * with only ~8 bytes/vector heap overhead (sortedIds + biases arrays).
 */
public class BipartiteReorderStrategy implements VectorReorderStrategy {

    private final VectorSimilarityFunction similarityFunction;

    public BipartiteReorderStrategy() {
        this(VectorSimilarityFunction.EUCLIDEAN);
    }

    public BipartiteReorderStrategy(VectorSimilarityFunction similarityFunction) {
        this.similarityFunction = similarityFunction;
    }

    @Override
    public int[] computePermutation(FloatVectorValues vectors, int numThreads) throws IOException {
        BpVectorReorderer reorderer = new BpVectorReorderer("vectors");
        reorderer.setMinPartitionSize(1);

        ForkJoinPool pool = new ForkJoinPool(numThreads);
        try {
            TaskExecutor executor = new TaskExecutor(pool);
            Sorter.DocMap map = reorderer.computeValueMap(vectors, similarityFunction, executor);

            int n = vectors.size();
            int[] permutation = new int[n];
            for (int i = 0; i < n; i++) {
                permutation[i] = map.newToOld(i);
            }
            return permutation;
        } finally {
            pool.shutdown();
        }
    }
}
