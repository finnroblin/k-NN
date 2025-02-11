#include <jni.h>
#include <cmath>

extern "C" {


// Note: l2 squared distance only for greater/less comparisons (no need to sqrt)
JNIEXPORT jfloat JNICALL Java_org_opensearch_knn_jni_luceneNativeService_l2SquaredNative
  (JNIEnv *env, jclass cls, jfloatArray queryVector, jfloatArray inputVector) {

    jfloat *queryArr = env->GetFloatArrayElements(queryVector, NULL);
    jfloat *inputArr = env->GetFloatArrayElements(inputVector, NULL);
    jsize length = env->GetArrayLength(queryVector);

    float sum = 0.0f;
    for (int i = 0; i < length; i++) {
        float diff = queryArr[i] - inputArr[i];
        sum += diff * diff;
    }

    env->ReleaseFloatArrayElements(queryVector, queryArr, JNI_ABORT);
    env->ReleaseFloatArrayElements(inputVector, inputArr, JNI_ABORT);

    return sum;
}

}

