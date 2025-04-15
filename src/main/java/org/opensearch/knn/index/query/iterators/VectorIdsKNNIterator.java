/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.iterators;

import org.apache.lucene.search.DocIdSetIterator;
import org.opensearch.common.Nullable;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.query.SegmentLevelQuantizationInfo;
import org.opensearch.knn.index.query.SegmentLevelQuantizationUtil;
import org.opensearch.knn.index.vectorvalues.KNNFloatVectorValues;

import lombok.extern.log4j.Log4j2;

import java.io.IOException;

/**
 * Inspired by DiversifyingChildrenFloatKnnVectorQuery in lucene
 * https://github.com/apache/lucene/blob/7b8aece125aabff2823626d5b939abf4747f63a7/lucene/join/src/java/org/apache/lucene/search/join/DiversifyingChildrenFloatKnnVectorQuery.java#L162
 *
 * The class is used in KNNWeight to score all docs, but, it iterates over filterIdsArray if filter is provided
 */
@Log4j2
public class VectorIdsKNNIterator implements KNNIterator {
    protected final DocIdSetIterator filterIdsIterator;
    protected final float[] queryVector;
    private final byte[] quantizedQueryVector;
    protected final KNNFloatVectorValues knnFloatVectorValues;
    protected final SpaceType spaceType;
    protected float currentScore = Float.NEGATIVE_INFINITY;
    protected int docId;
    private final SegmentLevelQuantizationInfo segmentLevelQuantizationInfo;

    public VectorIdsKNNIterator(
        @Nullable final DocIdSetIterator filterIdsIterator,
        final float[] queryVector,
        final KNNFloatVectorValues knnFloatVectorValues,
        final SpaceType spaceType
    ) throws IOException {
        this(filterIdsIterator, queryVector, knnFloatVectorValues, spaceType, null, null);
    }

    public VectorIdsKNNIterator(final float[] queryVector, final KNNFloatVectorValues knnFloatVectorValues, final SpaceType spaceType)
        throws IOException {
        this(null, queryVector, knnFloatVectorValues, spaceType, null, null);
    }

    public VectorIdsKNNIterator(
        @Nullable final DocIdSetIterator filterIdsIterator,
        final float[] queryVector,
        final KNNFloatVectorValues knnFloatVectorValues,
        final SpaceType spaceType,
        final byte[] quantizedQueryVector,
        final SegmentLevelQuantizationInfo segmentLevelQuantizationInfo
    ) throws IOException {
        this.filterIdsIterator = filterIdsIterator;
        this.queryVector = queryVector;
        this.knnFloatVectorValues = knnFloatVectorValues;
        this.spaceType = spaceType;
        // This cannot be moved inside nextDoc() method since it will break when we have nested field, where
        // nextDoc should already be referring to next knnVectorValues
        this.docId = getNextDocId();
        this.quantizedQueryVector = quantizedQueryVector;
        this.segmentLevelQuantizationInfo = segmentLevelQuantizationInfo;
    }

    /**
     * Advance to the next doc and update score value with score of the next doc.
     * DocIdSetIterator.NO_MORE_DOCS is returned when there is no more docs
     *
     * @return next doc id
     */
    @Override
    public int nextDoc() throws IOException {

        if (docId == DocIdSetIterator.NO_MORE_DOCS) {
            return DocIdSetIterator.NO_MORE_DOCS;
        }
        currentScore = computeScore();
        int currentDocId = docId;
        docId = getNextDocId();
        return currentDocId;
    }

    @Override
    public float score() {
        return currentScore;
    }

    protected float computeScore() throws IOException {
        final float[] vector = knnFloatVectorValues.getVector();

        /*
        * do some more investigation for rescoring... 
        rescore on -> computeScore called, should hit else block. 
            * add 
            * filter and for exact search (threshold is low, doesn't build graph strucutre), we do exact search on the index.
            efficient filtering -- serach idx w filter, if hnsw level is super sparse then do an exact search.

            faiss will return hamming distance codes, for SEGMENT CONSISTENCY we need to use exact search on HAMMING 
        */        
        if (segmentLevelQuantizationInfo != null) {
            // TODO here
            if (SegmentLevelQuantizationUtil.isAdcEnabled(segmentLevelQuantizationInfo)) {
                log.info("ComptueScore called w vector at position 0: " + vector[0] + " and queryvector at pos 0: " + queryVector[0]);
                double distance = 0.0f;
                for (int i = 0; i < vector.length; i++) {
                    // TODO: This only makes sense for l2
                    distance += Math.pow(vector[i] - queryVector[i], 2);
                }
                return (float) distance;
            }
            // redo the hamming so that we can stay consistent w the faiss scores.
            byte[] quantizedVector = SegmentLevelQuantizationUtil.quantizeVector(vector, segmentLevelQuantizationInfo);
            return SpaceType.HAMMING.getKnnVectorSimilarityFunction().compare(quantizedQueryVector, quantizedVector);
        }
        // Calculates a similarity score between the two vectors with a specified function. Higher similarity
        // scores correspond to closer vectors.
        // note: the query vector is not transformed here if using adc, which I think is expected.
        return spaceType.getKnnVectorSimilarityFunction().compare(queryVector, vector);
//        double distance = 0.0f;
//        for (int i = 0; i < vector.length; i++) {
//            // TODO: This only makes sense for l2
//            distance += Math.pow(vector[i] - queryVector[i], 2);
//        }
//        return (float) distance;
        // log.info("just doing the vector calc again" + vector[0] + " .  " + vector[1]);
        
        
        // double distance = 0.0f;
        // for (int i = 0; i < vector.length; i++) {
        //         // TODO: This only makes sense for l2
        //         distance += Math.pow(vector[i] - queryVector[i], 2);
        // }
        // return (float) distance;
        
    }

    protected int getNextDocId() throws IOException {
        if (filterIdsIterator == null) {
            return knnFloatVectorValues.nextDoc();
        }
        int nextDocID = this.filterIdsIterator.nextDoc();
        // For filter case, advance vector values to corresponding doc id from filter bit set
        if (nextDocID != DocIdSetIterator.NO_MORE_DOCS) {
            knnFloatVectorValues.advance(nextDocID);
        }
        return nextDocID;
    }
}
