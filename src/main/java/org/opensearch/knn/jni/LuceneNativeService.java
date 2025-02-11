/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.jni;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.opensearch.knn.plugin.script.KNNScoringUtil;

import java.security.AccessController;
import java.security.PrivilegedAction;

public class LuceneNativeService {
    static {

        AccessController.doPrivileged((PrivilegedAction<Void>) () -> {
            System.loadLibrary("opensearchknn_lucene_knn");
            return null;
        });
    }
    public static native float l2SquaredNative(float[] queryVector, float[] inputVector);
}
