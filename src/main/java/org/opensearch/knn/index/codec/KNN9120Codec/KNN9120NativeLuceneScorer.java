/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN9120Codec;

import org.apache.lucene.codecs.hnsw.FlatVectorsScorer;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.util.hnsw.RandomAccessVectorValues;
import org.apache.lucene.util.hnsw.RandomVectorScorer;
import org.apache.lucene.util.hnsw.RandomVectorScorerSupplier;
import org.opensearch.knn.index.KNNVectorSimilarityFunction;

import java.io.IOException;

public class KNN9120NativeLuceneScorer implements FlatVectorsScorer {

    @Override
    public RandomVectorScorerSupplier getRandomVectorScorerSupplier(VectorSimilarityFunction similarityFunction, RandomAccessVectorValues vectorValues) throws IOException {
        return new NativeLuceneRandomVectorScorerSupplier((RandomAccessVectorValues.Floats) vectorValues);
    }

    @Override
    public RandomVectorScorer getRandomVectorScorer(VectorSimilarityFunction similarityFunction, RandomAccessVectorValues vectorValues, float[] target) throws IOException {
        return new NativeLuceneVectorScorer((RandomAccessVectorValues.Floats) vectorValues, target);
    }

    @Override
    public RandomVectorScorer getRandomVectorScorer(VectorSimilarityFunction similarityFunction, RandomAccessVectorValues vectorValues, byte[] target) throws IOException {
        throw new IllegalArgumentException("native lucene vectors do not support byte[] targets");
    }

    static class NativeLuceneVectorScorer implements RandomVectorScorer {
        private final RandomAccessVectorValues.Floats vectorValues;
        private final float[] queryVector;
        NativeLuceneVectorScorer(RandomAccessVectorValues.Floats vectorValues, float[] query) {
            this.queryVector = query;
            this.vectorValues = vectorValues;
        }

        @Override
        public float score(int node) throws IOException {
            return KNNVectorSimilarityFunction.EUCLIDEAN.compare(queryVector, vectorValues.vectorValue(node));
        }

        @Override
        public int maxOrd() {
            return vectorValues.size();
        }

        @Override
        public int ordToDoc(int ord) {
            return vectorValues.ordToDoc(ord);
        }

    }

    static class NativeLuceneRandomVectorScorerSupplier implements RandomVectorScorerSupplier {
        protected final RandomAccessVectorValues.Floats vectorValues;
        protected final RandomAccessVectorValues.Floats vectorValues1;
        protected final RandomAccessVectorValues.Floats vectorValues2;

        public NativeLuceneRandomVectorScorerSupplier(RandomAccessVectorValues.Floats vectorValues) throws IOException {
            this.vectorValues = vectorValues;
            this.vectorValues1 = vectorValues.copy();
            this.vectorValues2 = vectorValues.copy();
        }

        @Override
        public RandomVectorScorer scorer(int ord) throws IOException {
            float[] queryVector = vectorValues1.vectorValue(ord);
            return new KNN9120NativeLuceneScorer.NativeLuceneVectorScorer(vectorValues2, queryVector);
        }

        @Override
        public RandomVectorScorerSupplier copy() throws IOException {
            return new KNN9120NativeLuceneScorer.NativeLuceneRandomVectorScorerSupplier(vectorValues.copy());
        }
    }
}
