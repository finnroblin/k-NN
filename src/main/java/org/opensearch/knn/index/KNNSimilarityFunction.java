/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index;

public interface KNNSimilarityFunction {
    float compare(float[] v1, float[] v2);
    float compare(byte[] v1, byte[] v2);
}
