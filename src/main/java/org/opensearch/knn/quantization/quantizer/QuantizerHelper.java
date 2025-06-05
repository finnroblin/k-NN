/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.quantization.quantizer;

import lombok.extern.log4j.Log4j2;
import org.opensearch.knn.quantization.models.quantizationParams.ScalarQuantizationParams;
import org.opensearch.knn.quantization.models.quantizationState.MultiBitScalarQuantizationState;
import org.opensearch.knn.quantization.models.quantizationState.OneBitScalarQuantizationState;
import org.opensearch.knn.quantization.models.requests.TrainingRequest;
import lombok.experimental.UtilityClass;
import oshi.util.tuples.Pair;

import java.io.IOException;

import static java.lang.Math.abs;

/**
 * Utility class for calculating quantization state information for both
 * OneBit and MultiBit scalar quantizers. Handles computing thresholds,
 * below/above mean statistics, and rotation matrix application.
 */
@UtilityClass
@Log4j2
class QuantizerHelper {
    private static final int ONE_BIT_BITS_PER_COORDINATE = 1;

    /**
     * Calculate quantization state for a {@link OneBitScalarQuantizationState}.
     *
     * @param trainingRequest      The training request containing the vectors.
     * @param sampledIndices       Sampled vector indices.
     * @param quantizationParams   Scalar quantization parameters.
     * @return A fully constructed {@link OneBitScalarQuantizationState}.
     * @throws IOException If vector retrieval fails.
     */
    static OneBitScalarQuantizationState calculateQuantizationState(
        TrainingRequest<float[]> trainingRequest,
        int[] sampledIndices,
        ScalarQuantizationParams quantizationParams
    ) throws IOException {
        QuantizerHelperResult quantizerHelperResult = calculateQuantizationStateHelper(
            trainingRequest,
            sampledIndices,
            ONE_BIT_BITS_PER_COORDINATE
        );

        return OneBitScalarQuantizationState.builder()
            .quantizationParams(quantizationParams)
            .meanThresholds(quantizerHelperResult.thresholds()[0])
            .rotationMatrix(quantizerHelperResult.rotationMatrix())
            .build();
    }

    /**
     * Calculate quantization state for a {@link MultiBitScalarQuantizationState}.
     *
     * @param trainingRequest      The training request containing vectors.
     * @param sampledIndices       Sampled vector indices.
     * @param quantizationParams   Scalar quantization parameters.
     * @param bitsPerCoordinate    Number of bits per dimension for quantization.
     * @return A fully constructed {@link MultiBitScalarQuantizationState}.
     * @throws IOException If vector retrieval fails.
     */
    static MultiBitScalarQuantizationState calculateQuantizationState(
        TrainingRequest<float[]> trainingRequest,
        int[] sampledIndices,
        ScalarQuantizationParams quantizationParams,
        int bitsPerCoordinate
    ) throws IOException {
        QuantizerHelperResult quantizerHelperResult = calculateQuantizationStateHelper(trainingRequest, sampledIndices, bitsPerCoordinate);

        return MultiBitScalarQuantizationState.builder()
            .quantizationParams(quantizationParams)
            .thresholds(quantizerHelperResult.thresholds())
            .rotationMatrix(quantizerHelperResult.rotationMatrix())
            .build();
    }

    /**
     * Validates that sampled indices are not null or empty.
     *
     * @param sampledIndices Indices to validate.
     */
    private static void validateSampledIndices(int[] sampledIndices) {
        if (sampledIndices == null || sampledIndices.length == 0) {
            throw new IllegalArgumentException("Sampled indices cannot be null or empty.");
        }
    }

    /**
     * Calculates thresholds used for multi-bit quantization.
     *
     * @param mean              Mean of each dimension.
     * @param stdDev            Standard deviation per dimension.
     * @param bitsPerCoordinate Number of bits per coordinate.
     * @return 2D array of thresholds of shape [bits][dimensions].
     */
    private static float[][] calculateThresholds(float[] mean, float[] stdDev, int bitsPerCoordinate, boolean isNormalized) {
        int dim = mean.length;
        float[][] thresholds = new float[bitsPerCoordinate][dim];
        float coef = bitsPerCoordinate + 1;

        for (int b = 0; b < bitsPerCoordinate; b++) {
            float iCoef = -1 + 2 * (b + 1) / coef;
            for (int d = 0; d < dim; d++) {
                if (isNormalized) {
                    // then threshold should only depend on standard deviation.
                    thresholds[b][d] = iCoef * stdDev[d];
                } else {
                    thresholds[b][d] = mean[d] + iCoef * stdDev[d];
                }
            }
        }
        return thresholds;
    }

    private record QuantizerHelperResult(float[][] rotationMatrix, float[][] thresholds) {
    }

    public record MeanAndStdDevResult(float[] mean, float[] stdDev, boolean isNormalized) {}

    private static QuantizerHelperResult calculateQuantizationStateHelper(
        TrainingRequest<float[]> trainingRequest,
        int[] sampledIndices,
        Integer bitsPerCoordinate  // 1 for one-bit, >1 for multi-bit
    ) throws IOException {
        validateSampledIndices(sampledIndices);
        int dim = trainingRequest.getVectorAtThePosition(sampledIndices[0]).length;

        float[][] rotationMatrix = null;
        if (trainingRequest.isEnableRandomRotation()) {
            rotationMatrix = RandomGaussianRotation.generateRotationMatrix(dim);
        }

        float[][] thresholds;

        MeanAndStdDevResult meanStd = calculateMeanAndStdDev(trainingRequest, sampledIndices, rotationMatrix);
        thresholds = calculateThresholds(meanStd.mean, meanStd.stdDev, bitsPerCoordinate, meanStd.isNormalized);
        // if bitsPerCoordinate = 1, there should only be one threshold (used to mean center coordinates).
        assert bitsPerCoordinate != 1 || thresholds.length == 1;

        return new QuantizerHelperResult(rotationMatrix, thresholds);
    }

    /**
     * Calculates per-dimension mean and standard deviation.
     *
     * @param request         Training request.
     * @param sampledIndices  Sampled vector indices.
     * @return Pair of (means[], stdDevs[]).
     * @throws IOException if vector access fails.
     */
    public static MeanAndStdDevResult calculateMeanAndStdDev(
        TrainingRequest<float[]> request,
        int[] sampledIndices,
        float[][] rotationMatrix
    ) throws IOException {
        float[] mean = null, sumSq = null;

        // vacuously the empty sample is normalized. This will become false the first time we see a non-normalized vector.
        boolean isNormalized = true;
        float norm = 0.0f;
        request.resetVectorValues();
        for (int docId : sampledIndices) {
            float[] vector = request.getVectorAtThePosition(docId);

            if (vector == null) {
                throw new IllegalArgumentException("Vector at sampled index " + docId + " is null.");
            }

            if (rotationMatrix != null) {
                vector = RandomGaussianRotation.applyRotation(vector, rotationMatrix);
            }

            if (mean == null) {
                mean = new float[vector.length];
                sumSq = new float[vector.length];
            }


            for (int i = 0; i < vector.length; i++) {
                if (isNormalized) {
                    // only need to store vector norms if normalized...
                    norm += vector[i] * vector[i];
                }
                mean[i] += vector[i];
                sumSq[i] += vector[i] * vector[i];
            }

            // the first time we see a non-normalized vector we can conclude that the entire dataset is not noramlzied.
            // Then we can skip the extra calculation work in the future.
            //
            if (isNormalized) {
                if (abs(1.0f - norm) > 0.0001) {
                    isNormalized = false;
                }
                norm = 0;
            }
        }

        if (mean == null) {
            throw new IllegalStateException("Mean array should not be null after processing vectors.");
        }

        int n = sampledIndices.length;
        for (int i = 0; i < mean.length; i++) {
            mean[i] /= n;
            // equivalent to standard deviation via algebra
            sumSq[i] = (float) Math.sqrt((sumSq[i] / n) - (mean[i] * mean[i]));
        }

        // here deal with normalization
        log.info("Dataset is normalized: {}", isNormalized);
//        return new Pair<>(mean, sumSq);
        return new MeanAndStdDevResult(mean, sumSq, isNormalized);
    }
}
