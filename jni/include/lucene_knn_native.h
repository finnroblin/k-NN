#ifndef KNN_NATIVE_H
#define KNN_NATIVE_H

#include <jni.h>

#ifdef __cplusplus
extern "C" {
#endif

JNIEXPORT jfloat JNICALL Java_org_opensearch_knn_jni_luceneNativeService_l2SquaredNative
  (JNIEnv *, jclass, jfloatArray, jfloatArray);

#ifdef __cplusplus
}
#endif

#endif

