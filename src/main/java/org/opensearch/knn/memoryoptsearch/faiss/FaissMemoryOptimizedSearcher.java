/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch.faiss;

import org.apache.lucene.codecs.hnsw.FlatVectorsScorer;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.search.knn.KnnSearchStrategy;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.IOSupplier;
import org.apache.lucene.util.hnsw.HnswGraphSearcher;
import org.apache.lucene.util.hnsw.OrdinalTranslatedKnnCollector;
import org.apache.lucene.util.hnsw.RandomVectorScorer;
import org.opensearch.knn.index.KNNVectorSimilarityFunction;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.memoryoptsearch.VectorSearcher;
import org.opensearch.knn.memoryoptsearch.faiss.cagra.FaissCagraHNSW;

import java.io.IOException;
import java.util.concurrent.ThreadLocalRandom;

/**
 * This searcher directly reads FAISS index file via the provided {@link IndexInput} then perform vector search on it.
 */
public class FaissMemoryOptimizedSearcher implements VectorSearcher {
    private final IndexInput indexInput;
    private final FaissIndex faissIndex;
    private final FlatVectorsScorer flatVectorsScorer;
    private final FaissHNSW hnsw;
    private final VectorSimilarityFunction vectorSimilarityFunction;
    private final long fileSize;
    private boolean isAdc;
    private final FlatVectorsScorerProvider.ADCFlatVectorsScorer adcScorer;

    // could add extra isAdc constructor maybe? this is a mess though...
//    public FaissMemoryOptimizedSearcher(final IndexInput indexInput) throws IOException {
//        this.indexInput = indexInput;
//        this.fileSize = indexInput.length();
//        this.faissIndex = FaissIndex.load(indexInput);
//        final KNNVectorSimilarityFunction knnVectorSimilarityFunction = faissIndex.getVectorSimilarityFunction();
//        this.flatVectorsScorer = FlatVectorsScorerProvider.getFlatVectorsScorer(knnVectorSimilarityFunction);
//        if (knnVectorSimilarityFunction != KNNVectorSimilarityFunction.HAMMING) {
//            vectorSimilarityFunction = knnVectorSimilarityFunction.getVectorSimilarityFunction();
//        } else {
//            vectorSimilarityFunction = null;
//        }
//        this.hnsw = extractFaissHnsw(faissIndex);
////        this.isAdc = false;
////        this.adcScorer = null;
//    }

    public FaissMemoryOptimizedSearcher(final IndexInput indexInput, boolean isAdc) throws IOException {
//        this(indexInput);
        this.indexInput = indexInput;
        this.fileSize = indexInput.length();
        this.faissIndex = FaissIndex.load(indexInput);
        final KNNVectorSimilarityFunction knnVectorSimilarityFunction = faissIndex.getVectorSimilarityFunction();

        if (knnVectorSimilarityFunction != KNNVectorSimilarityFunction.HAMMING) {
            vectorSimilarityFunction = knnVectorSimilarityFunction.getVectorSimilarityFunction();
        } else {
            vectorSimilarityFunction = null;
        }

        if (isAdc) {
            this.adcScorer = FlatVectorsScorerProvider.getAdcFlatVectorScorer(knnVectorSimilarityFunction);
            this.flatVectorsScorer = null;
            this.isAdc = true;
        } else {
            this.adcScorer = null;
            this.flatVectorsScorer = FlatVectorsScorerProvider.getFlatVectorsScorer(knnVectorSimilarityFunction);
            this.isAdc = false;
        }
//        this.flatVectorsScorer = null;
//        this.adcScorer = FlatVectorsScorerProvider.getAdcFlatVectorScorer(knnVectorSimilarityFunction);
//        if (knnVectorSimilarityFunction != KNNVectorSimilarityFunction.HAMMING) {
//            vectorSimilarityFunction = knnVectorSimilarityFunction.getVectorSimilarityFunction();
//        } else {
//            vectorSimilarityFunction = null;
//        }
        this.hnsw = extractFaissHnsw(faissIndex);
        // TODO: it's really ugly to just defer to different constructors with the adc method/just by passing in a new boolean...
        // I think a lot of work might be needed to make this clean, but it'll be okay for a first draft.
//        this.isAdc = true;
//        this.flatVectorsScorer = FlatVectorsScorerProvider.getAdcFlatVectorScorer(v);
        // start as a
//        this.adcScorer = () -> FlatVectorsScorerProvider.ADCFlatVectorsScorer;


    }


        private static FaissHNSW extractFaissHnsw(final FaissIndex faissIndex) {
        if (faissIndex instanceof FaissIdMapIndex idMapIndex) {
            return idMapIndex.getFaissHnsw();
        }

        throw new IllegalArgumentException("Faiss index [" + faissIndex.getIndexType() + "] does not have HNSW as an index.");
    }

    public void searchWithAdc(float[] target, KnnCollector knnCollector, Bits acceptDocs, SpaceType spaceType) throws IOException {
        // can I maybe just inject the flatVectorsScorer here? let's do that as a first draft.
        // somehow need to override the flatVectorsScorer and grab it...
        search(VectorEncoding.FLOAT32,
                () -> adcScorer.getRandomVectorScorerForAdc(
                        vectorSimilarityFunction,
                        faissIndex.getByteValues(getSlicedIndexInput()),
                        target,
                        spaceType
                ),
                knnCollector,
                acceptDocs
        );
    }

    @Override
    public void search(float[] target, KnnCollector knnCollector, Bits acceptDocs) throws IOException {
        // TODO: below fails on the getFloatValues call... we need to pass to a different function based on if adc is enabled...
        // maybe a searchWithADC command like search(float32, geTRandomVectorScorerr on faissIndex.getByteValues(....
        // probably a way to hack this together.
        // maybe we want to query a binary index?
//        knnCollector doesn't have what we want.
//        search(
// needs to take in the target based on the
//        )
//        faissIndex.
        search(
                VectorEncoding.FLOAT32,
                () -> flatVectorsScorer.getRandomVectorScorer(
                        vectorSimilarityFunction,
                        faissIndex.getFloatValues(getSlicedIndexInput()),
                        target
                ),
                knnCollector,
                acceptDocs
        );
//        search(
//            VectorEncoding.FLOAT32,
//            () -> flatVectorsScorer.getRandomVectorScorer(
//                vectorSimilarityFunction,
//                faissIndex.getFloatValues(getSlicedIndexInput()),
//                target
//            ), // scorerSupplier is here. We need to essentially load the binary index, and then use a custom
//                // scorer supplier that supports it. But I don't know how to get the index from the knnColector.
//            knnCollector,
//            acceptDocs
//        );
    }

    @Override
    public void search(byte[] target, KnnCollector knnCollector, Bits acceptDocs) throws IOException {
        search(
            VectorEncoding.BYTE,
            () -> flatVectorsScorer.getRandomVectorScorer(
                vectorSimilarityFunction,
                faissIndex.getByteValues(getSlicedIndexInput()),
                target
            ),
            knnCollector,
            acceptDocs
        );
    }

    @Override
    public void close() throws IOException {
        indexInput.close();
    }

    // only difference is with the scorer, so that's the only one to do.
    private void searchWithADC(
            final VectorEncoding vectorEncoding,
            final IOSupplier<RandomVectorScorer> scorerSupplier,
            final KnnCollector knnCollector,
            final Bits acceptDocs // here need to see if it can be modified in both usages.
    ) throws IOException {
        // Set up required components for vector search
        final RandomVectorScorer scorer = scorerSupplier.get();
        final KnnCollector collector = createKnnCollector(knnCollector, scorer);
        final Bits acceptedOrds = scorer.getAcceptOrds(acceptDocs);

        if (knnCollector.k() < scorer.maxOrd()) {
            // Do ANN search with Lucene's HNSW graph searcher.
            HnswGraphSearcher.search(scorer, collector, new FaissHnswGraph(hnsw, getSlicedIndexInput()), acceptedOrds);
        } else {
            // If k is larger than the number of vectors, we can just iterate over all vectors
            // and collect them.
            for (int i = 0; i < scorer.maxOrd(); i++) {
                if (acceptedOrds == null || acceptedOrds.get(i)) {
                    if (!knnCollector.earlyTerminated()) {
                        knnCollector.incVisitedCount(1);
                        knnCollector.collect(scorer.ordToDoc(i), scorer.score(i));
                    } else {
                        break;
                    }
                }
            }
        }  // End if
    }

    private void search(
        final VectorEncoding vectorEncoding,
        final IOSupplier<RandomVectorScorer> scorerSupplier,
        final KnnCollector knnCollector,
        final Bits acceptDocs // here need to see if it can be modified in both usages.
    ) throws IOException {
        if (faissIndex.getTotalNumberOfVectors() == 0 || knnCollector.k() == 0) {
            return;
        }
// TODO: in ADC it's okay if faissIndex.getVectorEncoding() is Byte but vectorEncoding is float
        if (false) {
//        if (faissIndex.getVectorEncoding() != vectorEncoding ) {
            throw new IllegalArgumentException(
                "Search for vector encoding ["
                    + vectorEncoding
                    + "] is not supported in "
                    + "an index vector whose encoding is ["
                    + faissIndex.getVectorEncoding()
                    + "]"
            );
        }

        // Set up required components for vector search
        final RandomVectorScorer scorer = scorerSupplier.get();
        final KnnCollector collector = createKnnCollector(knnCollector, scorer);
        final Bits acceptedOrds = scorer.getAcceptOrds(acceptDocs);

        if (knnCollector.k() < scorer.maxOrd()) {
            // Do ANN search with Lucene's HNSW graph searcher.
            HnswGraphSearcher.search(scorer, collector, new FaissHnswGraph(hnsw, getSlicedIndexInput()), acceptedOrds);
        } else {
            // If k is larger than the number of vectors, we can just iterate over all vectors
            // and collect them.
            for (int i = 0; i < scorer.maxOrd(); i++) {
                if (acceptedOrds == null || acceptedOrds.get(i)) {
                    if (!knnCollector.earlyTerminated()) {
                        knnCollector.incVisitedCount(1);
                        knnCollector.collect(scorer.ordToDoc(i), scorer.score(i));
                    } else {
                        break;
                    }
                }
            }
        }  // End if
    }

    private IndexInput getSlicedIndexInput() throws IOException {
        return indexInput.slice("FaissMemoryOptimizedSearcher", 0, fileSize);
    }

    private KnnCollector createKnnCollector(final KnnCollector knnCollector, final RandomVectorScorer scorer) {
        final KnnCollector ordinalTranslatedKnnCollector = new OrdinalTranslatedKnnCollector(knnCollector, scorer::ordToDoc);

        if (hnsw instanceof FaissCagraHNSW cagraHNSW) {
            return new KnnCollector.Decorator(ordinalTranslatedKnnCollector) {
                @Override
                public KnnSearchStrategy getSearchStrategy() {
                    return new RandomEntryPointsKnnSearchStrategy(
                        cagraHNSW.getNumBaseLevelSearchEntryPoints(),
                        cagraHNSW.getTotalNumberOfVectors(),
                        knnCollector.getSearchStrategy()
                    );
                }
            };
        } else {
            return ordinalTranslatedKnnCollector;
        }
    }

    /**
     * Knn search strategy having a doc-id-iterator returning random document ids.
     * This is not designed for general purpose, it is particularly designed for populating random document ids for Cagra index.
     * Note that doc-id-iterator returns a random ids in `nextDoc` method without sorting, and might return duplicated ids.
     */
    static class RandomEntryPointsKnnSearchStrategy extends KnnSearchStrategy.Seeded {
        public RandomEntryPointsKnnSearchStrategy(
            final int numberOfEntryPoints,
            final long totalNumberOfVectors,
            final KnnSearchStrategy originalStrategy
        ) {
            super(
                generateRandomEntryPoints(numberOfEntryPoints, Math.toIntExact(totalNumberOfVectors)),
                numberOfEntryPoints,
                originalStrategy
            );
        }

        private static DocIdSetIterator generateRandomEntryPoints(final int numberOfEntryPoints, int totalNumberOfVectors) {
            return new DocIdSetIterator() {
                int numPopulatedVectors = 0;

                @Override
                public int docID() {
                    throw new UnsupportedOperationException("DISI in RandomEntryPointsKnnSearchStrategy does not support docID()");
                }

                @Override
                public int nextDoc() {
                    if (numPopulatedVectors < numberOfEntryPoints) {
                        ++numPopulatedVectors;
                        // It is fine to populate the same doc ids here, the same vectors will not be visited more than once with bitset.
                        return ThreadLocalRandom.current().nextInt(totalNumberOfVectors);
                    }

                    return NO_MORE_DOCS;
                }

                @Override
                public int advance(int targetDoc) {
                    throw new UnsupportedOperationException("DISI in RandomEntryPointsKnnSearchStrategy does not support advance(int)");
                }

                @Override
                public long cost() {
                    throw new UnsupportedOperationException("DISI in RandomEntryPointsKnnSearchStrategy does not support cost()");
                }
            };
        }
    }
}
