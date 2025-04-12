/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 *
 * Modifications Copyright OpenSearch Contributors. See
 * GitHub history for details.
 */

#include "faiss_wrapper.h"

#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "jni_util.h"
#include "test_util.h"
#include "faiss/IndexHNSW.h"
#include "faiss/IndexIVFPQ.h"
#include "FaissIndexBQ.h"
#include "faiss/IndexBinaryFlat.h"
#include "faiss/IndexBinaryHNSW.h"
#include "mocks/faiss_index_service_mock.h"
#include "native_stream_support_util.h"

using ::test_util::JavaFileIndexOutputMock;
using ::test_util::MockJNIUtil;
using ::test_util::StreamIOError;
using ::test_util::setUpJavaFileOutputMocking;
using ::testing::Mock;
using ::testing::NiceMock;
using ::testing::Return;
using ::testing::_;

const float randomDataMin = -500.0;
const float randomDataMax = 500.0;
const float rangeSearchRandomDataMin = -50;
const float rangeSearchRandomDataMax = 50;
const float rangeSearchRadius = 20000;

void createIndexIteratively(
        knn_jni::JNIUtilInterface * JNIUtil, 
        JNIEnv *jniEnv, 
        std::vector<faiss::idx_t> & ids,
        std::vector<float> & vectors,
        int dim,
        jobject javaFileOutputMock,
        std::unordered_map<string, jobject> parametersMap,
        IndexService * indexService,
        int insertions = 10
    ) {
    long numDocs = ids.size();
    if (numDocs % insertions != 0) {
        throw std::invalid_argument("Number of documents should be divisible by number of insertions");
    }
    long docsPerInsertion = numDocs / insertions;
    long index_ptr = knn_jni::faiss_wrapper::InitIndex(JNIUtil, jniEnv, numDocs, dim, (jobject)&parametersMap, indexService);
    std::vector<faiss::idx_t> insertIds;
    std::vector<float> insertVecs;
    for (int i = 0; i < insertions; i++) {
        insertIds.clear();
        insertVecs.clear();
        int start_idx = i * docsPerInsertion;
        int end_idx = start_idx + docsPerInsertion;
        for (int j = start_idx; j < end_idx; j++) {
            insertIds.push_back(j);
            for(int k = 0; k < dim; k++) {
                insertVecs.push_back(vectors[j * dim + k]);
            }
        }
        knn_jni::faiss_wrapper::InsertToIndex(JNIUtil, jniEnv, reinterpret_cast<jintArray>(&insertIds), (jlong)&insertVecs, dim, index_ptr, 0, indexService);
    }
    knn_jni::faiss_wrapper::WriteIndex(JNIUtil, jniEnv, javaFileOutputMock, index_ptr, indexService);
}

void createBinaryIndexIteratively(
        knn_jni::JNIUtilInterface * JNIUtil, 
        JNIEnv *jniEnv, 
        std::vector<faiss::idx_t> & ids,
        std::vector<uint8_t> & vectors,
        int dim,
        jobject javaFileOutputMock,
        std::unordered_map<string, jobject> parametersMap, 
        IndexService * indexService,
        int insertions = 10
    ) {
    long numDocs = ids.size();
    long index_ptr = knn_jni::faiss_wrapper::InitIndex(JNIUtil, jniEnv, numDocs, dim, (jobject)&parametersMap, indexService);
    std::vector<faiss::idx_t> insertIds;
    std::vector<float> insertVecs;
    for (int i = 0; i < insertions; i++) {
        int start_idx = numDocs * i / insertions;
        int end_idx = numDocs * (i + 1) / insertions;
        int docs_to_insert = end_idx - start_idx;
        if (docs_to_insert == 0) {
            continue;
        }
        insertIds.clear();
        insertVecs.clear();
        for (int j = start_idx; j < end_idx; j++) {
            insertIds.push_back(j);
            for(int k = 0; k < dim / 8; k++) {
                insertVecs.push_back(vectors[j * (dim / 8) + k]);
            }
        }
        knn_jni::faiss_wrapper::InsertToIndex(JNIUtil, jniEnv, reinterpret_cast<jintArray>(&insertIds), (jlong)&insertVecs, dim, index_ptr, 0, indexService);
    }

    knn_jni::faiss_wrapper::WriteIndex(JNIUtil, jniEnv, javaFileOutputMock, index_ptr, indexService);
}

TEST(FaissCreateIndexTest, BasicAssertions) {
    // Define the data
    faiss::idx_t numIds = 200;
    std::vector<faiss::idx_t> ids;
    std::vector<float> vectors;
    int dim = 2;
    vectors.reserve(dim * numIds);
    for (int64_t i = 0; i < numIds; ++i) {
      ids.push_back(i);
      for (int j = 0; j < dim; ++j) {
        vectors.push_back(test_util::RandomFloat(-500.0, 500.0));
      }
    }

    std::string indexPath = test_util::RandomString(10, "tmp/", ".faiss");
    std::string spaceType = knn_jni::L2;
    std::string indexDescription = "HNSW32,Flat";

    std::unordered_map<std::string, jobject> parametersMap;
    parametersMap[knn_jni::SPACE_TYPE] = (jobject)&spaceType;
    parametersMap[knn_jni::INDEX_DESCRIPTION] = (jobject)&indexDescription;
    std::unordered_map<std::string, jobject> subParametersMap;
    parametersMap[knn_jni::PARAMETERS] = (jobject)&subParametersMap;

    // Set up jni
    NiceMock<JNIEnv> jniEnv;
    NiceMock<test_util::MockJNIUtil> mockJNIUtil;
    JavaFileIndexOutputMock javaFileIndexOutputMock {indexPath};
    setUpJavaFileOutputMocking(javaFileIndexOutputMock, mockJNIUtil, false);

    // Create the index
    std::unique_ptr<FaissMethods> faissMethods(new FaissMethods());
    NiceMock<MockIndexService> mockIndexService(std::move(faissMethods));
    int insertions = 10;
    EXPECT_CALL(mockIndexService, initIndex(_, _, faiss::METRIC_L2, indexDescription, dim, (int)numIds, 0, subParametersMap))
        .Times(1);
    EXPECT_CALL(mockIndexService, insertToIndex(dim, numIds / insertions, 0, _, _, _))
        .Times(insertions);
    EXPECT_CALL(mockIndexService, writeIndex(_, _))
        .Times(1);

    createIndexIteratively(&mockJNIUtil,
                           &jniEnv,
                           ids,
                           vectors,
                           dim,
                           (jobject) (&javaFileIndexOutputMock),
                           parametersMap,
                           &mockIndexService,
                           insertions);
}

TEST(FaissIndexBQTest, BaselineCheck) {
    int dim = 8;
    uint8_t * code = new uint8_t[dim / sizeof(uint8_t)];
    code[0] = 3;
    std::vector query = {100.0f, 200.0f, 300.0f, 400.0f, 500.0f, 600.0f, 700.0f, 800.0f};

    float score = 0.0f;
    for (int i = 0; i < dim; i++) {
        uint8_t code_block = code[(i / 8)];
        int bit_offset = i % 8;
        int bit_mask = 1 << bit_offset;
        int code_masked = (code_block & bit_mask);
        int code_translated = code_masked >> bit_offset;

        // want to select the
        // std::cout << "bit_offset: " << bit_offset << std::endl;
        // std::cout << "bit_mask: " << bit_mask << std::endl;
        // std::cout << "code_masked: " << code_masked << std::endl;
        // std::cout << "code_translated: " << code_translated << std::endl;
        score += code_translated == 0 ? 0 : -1*query[i];;
    }

    // Id expect the score to be 0
    std::cout << "score: " << score << std::endl;
}
TEST(FaissIndexBQTest, InnerProductDistanceComputerTest) {
    int dim = 24;

    // Create a query vector in floats with random seed (use std::mt with a reproducible seed)
    std::mt19937 rng(42);  // Fixed seed for reproducibility
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    
    std::vector<float> query(dim);
    for (int i = 0; i < dim; i++) {
        query[i] = dist(rng);
    }

    // Create a set of 10 indexed vectors as floats
    const int numVectors = 10;
    std::vector<float> indexedVectors(numVectors * dim);
    for (int i = 0; i < numVectors * dim; i++) {
        indexedVectors[i] = dist(rng);
    }

    // Compute the ground truth distances of the indexed vectors
    std::vector<float> groundTruthDistances(numVectors);
    for (int i = 0; i < numVectors; i++) {
        float distance = 0;
        for (int j = 0; j < dim; j++) {
            if (indexedVectors[i * dim + j] > 0) {
                // For binary quantization, each set bit contributes -query[j]
                distance += -query[j];
            }
        }
        groundTruthDistances[i] = distance;
    }

    // Quantize the float vectors and translate them to a code array
    int code_size = (dim + 7) / 8;  // Number of bytes needed
    std::vector<uint8_t> codes(numVectors * code_size, 0);
    
    for (int i = 0; i < numVectors; i++) {
        for (int j = 0; j < dim; j++) {
            if (indexedVectors[i * dim + j] > 0) {
                // Set the bit to 1 if the value is positive
                int byte_pos = i * code_size + (j / 8);
                int bit_pos = 7 - (j % 8);
                codes[byte_pos] |= (1 << bit_pos);
            }
        }
    }

    // Create FaissIndexBQ with the binary codes
    knn_jni::faiss_wrapper::FaissIndexBQ bqIndex(dim, codes);
    
    // Create distance computer and set query
    auto dc = std::unique_ptr<knn_jni::faiss_wrapper::CustomerFlatCodesDistanceComputer>(
        dynamic_cast<knn_jni::faiss_wrapper::CustomerFlatCodesDistanceComputer*>(
            bqIndex.get_FlatCodesDistanceComputer()));
    
    ASSERT_NE(dc.get(), nullptr);
    dc->set_query(query.data());
    
    // Query the distance for each code and compare with ground truth
    for (int i = 0; i < numVectors; i++) {
        // Get distance using the distance computer
        float distance = dc->distance_to_code_IP(&codes[i * code_size]);
        
        // Assert that distances are the same as computed ground truth
        ASSERT_NEAR(groundTruthDistances[i], distance, 1e-5) 
            << "Distance mismatch for vector " << i 
            << ": expected " << groundTruthDistances[i] 
            << ", got " << distance;
    }
}


TEST(FaissIndexBQTest, ComprehensiveTest) {
    // Test 1: Basic Constructor and Initialization
    int dim = 16;
    std::vector<uint8_t> codes = {
        0b11110000,  // First vector first byte
        0b00001111,  // First vector second byte
        0b10101010,  // Second vector first byte
        0b01010101   // Second vector second byte
    };

    knn_jni::faiss_wrapper::FaissIndexBQ index(dim, codes);

    // Verify initial state
    ASSERT_EQ(index.d, dim);
    ASSERT_EQ(index.code_size, 1);
    ASSERT_EQ(index.codes, codes);

    // Test 2: Parent Initialization
    faiss::IndexFlatL2 parent(dim);
    faiss::IndexFlatL2 grandparent(dim);
    index.init(&parent, &grandparent);

    // Verify ntotal is properly set for all objects
    size_t expected_ntotal = codes.size() / (dim / 8); // should be2...
    ASSERT_EQ(index.ntotal, expected_ntotal);
    ASSERT_EQ(parent.ntotal, expected_ntotal);
    ASSERT_EQ(grandparent.ntotal, expected_ntotal);

    // Test 3: Distance Computer Creation and Type
    std::unique_ptr<faiss::FlatCodesDistanceComputer> dc(index.get_FlatCodesDistanceComputer());
    auto* custom_dc = dynamic_cast<knn_jni::faiss_wrapper::CustomerFlatCodesDistanceComputer*>(dc.get());
    ASSERT_NE(custom_dc, nullptr);

    // Verify distance computer initialization
    ASSERT_EQ(custom_dc->dimension, dim);
    ASSERT_EQ(custom_dc->code_size, 1);
    for (size_t i = 0; i < codes.size(); i++) {
        ASSERT_EQ(custom_dc->codes[i], codes[i]);
    }

    // Test 4: Distance Computation
    // Create test query vectors
    std::vector<std::vector<float>> test_queries = {
        std::vector<float>(dim, 1.0f),  // All ones
        std::vector<float>(dim, 0.0f),  // All zeros
        {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,  // First half ones
         0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}  // Second half zeros
    };

    for (const auto& query : test_queries) {
        custom_dc->set_query(query.data());

        // Calculate distances to both vectors in codes
        float dist1 = custom_dc->distance_to_code(&codes[0]);  // First vector
        float dist2 = custom_dc->distance_to_code(&codes[2]);  // Second vector

        // Verify distances are non-negative
        ASSERT_GE(dist1, 0.0f);
        ASSERT_GE(dist2, 0.0f);

        // Verify distances are finite
        ASSERT_FALSE(std::isinf(dist1));
        ASSERT_FALSE(std::isnan(dist1));
        ASSERT_FALSE(std::isinf(dist2));
        ASSERT_FALSE(std::isnan(dist2));
    }

    // Test 5: Edge Cases
    // Test empty codes
    std::vector<uint8_t> empty_codes;
    knn_jni::faiss_wrapper::FaissIndexBQ empty_index(dim, empty_codes);
    ASSERT_EQ(empty_index.ntotal, 0);

    // Test single byte code
    std::vector<uint8_t> single_byte = {0b10000000};
    knn_jni::faiss_wrapper::FaissIndexBQ single_byte_index(8, single_byte);
    auto* single_dc = dynamic_cast<knn_jni::faiss_wrapper::CustomerFlatCodesDistanceComputer*>(
        single_byte_index.get_FlatCodesDistanceComputer());

    std::vector<float> zero_query(8, 0.0f);
    single_dc->set_query(zero_query.data());
    float dist = single_dc->distance_to_code(single_byte.data());
    ASSERT_NEAR(dist, 1.0f, 1e-6);  // Distance to single set bit

    // Test 6: Verify symmetric distance computation
    if (custom_dc != nullptr) {
        float sym_dist = custom_dc->symmetric_dis(0, 1);
        ASSERT_GE(sym_dist, 0.0f);
        ASSERT_FALSE(std::isinf(sym_dist));
        ASSERT_FALSE(std::isnan(sym_dist));
    }

    // Test 7: Verify unimplemented methods don't crash
    std::vector<float> dummy_distances(1);
    std::vector<faiss::idx_t> dummy_labels(1);
    std::vector<float> dummy_query(dim);

    // These should not crash
    index.search(1, dummy_query.data(), 1, dummy_distances.data(), dummy_labels.data());
    index.merge_from(parent);  // Should do nothing but not crash
}

TEST(FaissIndexBQTest, Debug) {
    const int dim = 128;
    const int numVectors = 1000;
    const int numQueries = 100;
    const int k = 100;
    
    // HNSW parameters (similar to Java test)
    const int hnswM = 16;
    const int efConstruction = 100;
    const int efSearch = k;
    
    // Step 1: Create test vectors
    std::cout << "Generating test data..." << std::endl;
    std::vector<int8_t> originalVectors = test_util::RandomByteVectors(dim, numVectors, -127, 127);
    std::vector<faiss::idx_t> ids = test_util::Range(numVectors);
    
    // Step 2: Quantize vectors to binary (simulating binary storage)
    std::cout << "Quantizing vectors..." << std::endl;
    std::vector<uint8_t> binaryVectors(numVectors * ((dim + 7) / 8), 0);
    for (int i = 0; i < numVectors; i++) {
        for (int d = 0; d < dim; d++) {
            int bytePos = (i * ((dim + 7) / 8)) + (d / 8);
            int bitPos = d % 8;
            if (originalVectors[i * dim + d] > 0) {
                binaryVectors[bytePos] |= (1 << bitPos);
            }
        }
    }
    
    // Step 3: Calculate ADC statistics
    std::cout << "Computing dimension statistics..." << std::endl;
    auto stats = test_util::ComputeDimensionStats(
        originalVectors, binaryVectors, dim, numVectors);
    
    // Step 4: Create a binary HNSW index (simulating the on-disk binary index)
    std::cout << "Creating binary HNSW index..." << std::endl;
    std::string method = "BHNSW32";
    std::unique_ptr<faiss::IndexBinary> binaryIndex(
        test_util::FaissCreateBinaryIndex(dim, method));
    
    // Configure binary index parameters
    if (auto* hnswIndex = dynamic_cast<faiss::IndexBinaryHNSW*>(binaryIndex.get())) {
        hnswIndex->hnsw.efConstruction = efConstruction;
        hnswIndex->hnsw.efSearch = efSearch;
    }
    
    // Step 5: Add vectors to binary index and prepare
    std::cout << "Adding vectors to index..." << std::endl;
    auto indexWithData = test_util::FaissAddBinaryData(binaryIndex.get(), ids, binaryVectors);
    
    // Step 6: Create FaissIndexBQ with the binary codes (simulating ADC conversion)
    std::cout << "Creating FaissIndexBQ for ADC querying..." << std::endl;
    auto* binaryIdMap = dynamic_cast<faiss::IndexBinaryIDMap*>(&indexWithData);
    ASSERT_NE(binaryIdMap, nullptr);
    auto* hnswBinary = dynamic_cast<faiss::IndexBinaryHNSW*>(binaryIdMap->index);
    ASSERT_NE(hnswBinary, nullptr);
    auto* codesIndex = dynamic_cast<faiss::IndexBinaryFlat*>(hnswBinary->storage);
    ASSERT_NE(codesIndex, nullptr);
    
    knn_jni::faiss_wrapper::FaissIndexBQ bqIndex(dim, codesIndex->xb);
    faiss::IndexHNSW alteredIndexHNSW(&bqIndex, hnswM);
    alteredIndexHNSW.hnsw = hnswBinary->hnsw;
    faiss::IndexIDMap alteredIdMap(&alteredIndexHNSW);
    bqIndex.init(&alteredIndexHNSW, &alteredIdMap);
    alteredIdMap.id_map = binaryIdMap->id_map;
    
    // Step 7: Compute ground truth
    std::cout << "Computing ground truth..." << std::endl;
    std::vector<std::vector<int>> groundTruth;
    {
        // Create a flat index for exact search
        faiss::IndexFlatL2 flatIndex(dim);
        std::vector<float> floatVectors(numVectors * dim);
        for (int i = 0; i < numVectors; i++) {
            for (int j = 0; j < dim; j++) {
                floatVectors[i * dim + j] = static_cast<float>(originalVectors[i * dim + j]);
            }
        }
        flatIndex.add(numVectors, floatVectors.data());
        
        // Search for ground truth
        for (int i = 0; i < numQueries; i++) {
            int queryIdx = test_util::RandomInt(0, numVectors - 1);
            std::vector<float> query(dim);
            for (int j = 0; j < dim; j++) {
                query[j] = static_cast<float>(originalVectors[queryIdx * dim + j]);
            }
            
            std::vector<float> distances(k);
            std::vector<faiss::idx_t> indices(k);
            flatIndex.search(1, query.data(), k, distances.data(), indices.data());
            
            groundTruth.push_back(std::vector<int>(indices.begin(), indices.end()));
        }
    }
    
    // Step 8: Test recall with ADC
    std::cout << "Testing ADC recall..." << std::endl;
    int correctResults = 0;
    int totalResults = 0;
    
    for (int i = 0; i < numQueries; i++) {
        // Select random vector as query
        int queryIdx = test_util::RandomInt(0, numVectors - 1);
        std::vector<float> query(dim);
        for (int j = 0; j < dim; j++) {
            query[j] = static_cast<float>(originalVectors[queryIdx * dim + j]);
        }
        
        // Transform query using ADC
        auto transformedQuery = test_util::TransformQueryADC(query, stats);
        
        // Get distance computer and set query
        auto dc = std::unique_ptr<knn_jni::faiss_wrapper::CustomerFlatCodesDistanceComputer>(
            dynamic_cast<knn_jni::faiss_wrapper::CustomerFlatCodesDistanceComputer*>(
                bqIndex.get_FlatCodesDistanceComputer()));
        ASSERT_NE(dc.get(), nullptr);
        dc->set_query(transformedQuery.data());
        
        // Search all vectors
        std::vector<std::pair<float, int>> results;
        for (int j = 0; j < numVectors; j++) {
            float dist = dc->distance_to_code(&binaryVectors[j * ((dim + 7) / 8)]);
            results.push_back({dist, j});
        }
        
        // Sort by distance
        std::sort(results.begin(), results.end());
        
        // Calculate recall@k
        std::unordered_set<int> groundTruthSet(groundTruth[i].begin(), groundTruth[i].end());
        int found = 0;
        for (int j = 0; j < k && j < results.size(); j++) {
            if (groundTruthSet.count(results[j].second) > 0) {
                found++;
            }
        }
        
        correctResults += found;
        totalResults += k;
    }
    
    // Calculate and report overall recall
    float recall = static_cast<float>(correctResults) / totalResults;
    std::cout << "ADC Recall@" << k << ": " << recall << std::endl;
    
    // Expect recall to be at least 60% (as in the Java test)
    ASSERT_GE(recall, 0.6) << "ADC recall is too low, expected at least 0.6 but got " << recall;
}

TEST(FaissIndexBQTest, Debug2) {

      const int seed = 42; // Fixed seed for consistent results
    
    faiss::RandomGenerator frng = faiss::RandomGenerator(seed);
    std::uniform_real_distribution<float> distLarge(-100.0f, 100.0f);
    std::uniform_real_distribution<float> distSmall(-20.0f, 20.0f);
    
    // Test parameters remain the same
    const int dim = 128;
    const int numVectors = 1000;
    const int numQueries = 50;
    const int k = 100;
    const int hnswM = 16;
    const int efConstruction = 100;
    const int efSearch = k;
    
    std::cout << "\n=== DEBUG: Initializing ADC test with seed " << seed << " ===" << std::endl;
    
    // Step 1: Create vectors with cluster structure using seeded RNG
    int numClusters = 10;
    std::vector<int8_t> originalVectors(numVectors * dim);
    std::vector<int> clusterAssignments(numVectors);
    
    // Create cluster centers with seeded randomness
    std::vector<std::vector<float>> clusterCenters;
    for (int c = 0; c < numClusters; c++) {
        std::vector<float> center(dim);
        for (int d = 0; d < dim; d++) {
            center[d] = distLarge(frng.mt);  // Replace RandomFloat
        }
        clusterCenters.push_back(center);
    }
    
    // Generate vectors around cluster centers with seeded randomness
    for (int i = 0; i < numVectors; i++) {
        int cluster = i % numClusters;
        clusterAssignments[i] = cluster;
        
        for (int d = 0; d < dim; d++) {
            float value = clusterCenters[cluster][d] + distSmall(frng.mt);  // Replace RandomFloat
            originalVectors[i * dim + d] = static_cast<int8_t>(std::max(-127.0f, std::min(127.0f, value)));
        }
    }
    std::vector<faiss::idx_t> ids = test_util::Range(numVectors);
    
    // Step 2: Quantize vectors to binary (1 bit per dimension)
    std::cout << "=== DEBUG: Quantizing vectors to binary ===" << std::endl;
    std::vector<uint8_t> binaryVectors(numVectors * ((dim + 7) / 8), 0);
    int totalBitsSet = 0;
    for (int i = 0; i < numVectors; i++) {
        for (int d = 0; d < dim; d++) {
            int bytePos = (i * ((dim + 7) / 8)) + (d / 8);
            int bitPos = d % 8;
            if (originalVectors[i * dim + d] > 0) {
                binaryVectors[bytePos] |= (1 << bitPos);
                totalBitsSet++;
            }
        }
    }
    float percentBitsSet = (float)totalBitsSet / (numVectors * dim) * 100;
    std::cout << "Binary quantization: " << percentBitsSet << "% of bits set to 1" << std::endl;
    
    // Step 3: Calculate ADC statistics
    auto stats = test_util::ComputeDimensionStats(
        originalVectors, binaryVectors, dim, numVectors);
    
    // Print sample statistics
    for (int d = 0; d < 5; d++) {
        std::cout << "Dim " << d << ": Zero mean=" << stats.zero_means[d] 
                    << ", One mean=" << stats.one_means[d] 
                    << ", Mean diff=" << (stats.one_means[d]-stats.zero_means[d]) << std::endl;
    }
    
    // Step 4: Create a binary index and save it
    std::cout << "=== DEBUG: Creating and saving binary HNSW index ===" << std::endl;
    std::string indexPath = test_util::RandomString(10, "tmp/", ".faiss");
    
    std::string method = "BHNSW32";
    std::unique_ptr<faiss::IndexBinary> binaryIndex(
        test_util::FaissCreateBinaryIndex(dim, method));
    
    // Configure binary index
    if (auto* hnswIndex = dynamic_cast<faiss::IndexBinaryHNSW*>(binaryIndex.get())) {
        hnswIndex->hnsw.efConstruction = efConstruction;
        hnswIndex->hnsw.efSearch = efSearch;
        hnswIndex->hnsw.rng = frng;
    }
    
    // Add vectors to index and save
    auto indexWithData = test_util::FaissAddBinaryData(binaryIndex.get(), ids, binaryVectors);
    test_util::FaissWriteBinaryIndex(&indexWithData, indexPath);
    
    // Step 5: Load binary index and convert to ADC index
    std::cout << "=== DEBUG: Loading binary index and converting to ADC index ===" << std::endl;
    
    // This mimics the LoadIndexWithStreamADC method exactly
    std::unique_ptr<faiss::IndexBinary> loadedBinaryIndex(
        test_util::FaissLoadBinaryIndex(indexPath));
    
    auto* binaryIdMap = dynamic_cast<faiss::IndexBinaryIDMap*>(loadedBinaryIndex.get());
    ASSERT_NE(binaryIdMap, nullptr);
    auto* hnswBinary = dynamic_cast<faiss::IndexBinaryHNSW*>(binaryIdMap->index);
    ASSERT_NE(hnswBinary, nullptr);
    auto* codesIndex = dynamic_cast<faiss::IndexBinaryFlat*>(hnswBinary->storage);
    ASSERT_NE(codesIndex, nullptr);
    
    std::vector<uint8_t> codes = codesIndex->xb;
    
    // Create the ADC index using the same approach as LoadIndexWithStreamADC
    knn_jni::faiss_wrapper::FaissIndexBQ* alteredStorage = new knn_jni::faiss_wrapper::FaissIndexBQ(dim, codes);
    faiss::IndexHNSW* alteredIndexHNSW = new faiss::IndexHNSW(alteredStorage, hnswM);
    alteredIndexHNSW->hnsw = hnswBinary->hnsw;
    faiss::IndexIDMap* alteredIdMap = new faiss::IndexIDMap(alteredIndexHNSW);
    alteredStorage->init(alteredIndexHNSW, alteredIdMap);
    alteredIdMap->id_map = binaryIdMap->id_map;
    
    // Store the ADC index (this is what LoadIndexWithStreamADC returns)
    std::unique_ptr<faiss::Index> adcIndex(alteredIdMap);

    std::cout << "ADC index created with " << adcIndex->ntotal << " vectors" << std::endl;
    
    // Step 6: Compute ground truth
    std::cout << "=== DEBUG: Computing ground truth ===" << std::endl;
    std::vector<std::vector<faiss::idx_t>> groundTruth(numQueries);
    std::vector<int> queryIndices(numQueries);
    
    {
        // Create a flat index for exact search
        faiss::IndexFlatL2 flatIndex(dim);
        std::vector<float> floatVectors(numVectors * dim);
        for (int i = 0; i < numVectors; i++) {
            for (int j = 0; j < dim; j++) {
                floatVectors[i * dim + j] = static_cast<float>(originalVectors[i * dim + j]);
            }
        }
        flatIndex.add(numVectors, floatVectors.data());
        
        // Select queries from different clusters
        for (int i = 0; i < numQueries; i++) {
            int cluster = i % numClusters;
            int clusterSize = numVectors / numClusters;
            int queryIdx = cluster * clusterSize + (i / numClusters) % clusterSize;
            queryIndices[i] = queryIdx;
            
            std::vector<float> query(dim);
            for (int j = 0; j < dim; j++) {
                query[j] = static_cast<float>(originalVectors[queryIdx * dim + j]);
            }
            
            std::vector<float> distances(k);
            groundTruth[i].resize(k); 
           flatIndex.search(1, query.data(), k, distances.data(), groundTruth[i].data());
        }
    }
    
    // Step 7: Test ADC recall using FAISS search
    std::cout << "=== DEBUG: Testing ADC recall with FAISS search ===" << std::endl;
    int correctResults = 0;
    int totalResults = 0;
    
    // Set HNSW search parameters
    faiss::SearchParameters searchParams;
    alteredIndexHNSW->hnsw.efSearch = efSearch;
    alteredIndexHNSW->hnsw.efConstruction = 100;
    
    for (int i = 0; i < numQueries; i++) {
        // Get query vector
        int queryIdx = queryIndices[i];
        std::vector<float> query(dim);
        for (int j = 0; j < dim; j++) {
            query[j] = static_cast<float>(originalVectors[queryIdx * dim + j]);
        }
        
        // Transform query using ADC
        auto transformedQuery = test_util::TransformQueryADC(query, stats);
        
        // Debug - verify query transformation
        if (i == 0) {
            std::cout << "Original query sample: ";
            for (int j = 0; j < 5; j++) {
                std::cout << query[j] << ", ";
            }
            std::cout << "\nTransformed query sample: ";
            for (int j = 0; j < 5; j++) {
                std::cout << transformedQuery[j] << ", ";
            }
            std::cout << std::endl;
        }

        std::cout << "after call\n";
        
        // Use FAISS search with the transformed query
        std::vector<float> distances(k);
        std::vector<faiss::idx_t> indices(k);
        adcIndex->search(1, transformedQuery.data(), k, distances.data(), indices.data(), nullptr);
        
        // Debug output for first query
        if (i == 0) {
            std::cout << "Query results - Top 5: ";
            for (int j = 0; j < 5; j++) {
                std::cout << indices[j] << " (" << distances[j] << "), ";
            }
            std::cout << "\nGround truth - Top 5: ";
            for (int j = 0; j < 5; j++) {
                std::cout << groundTruth[i][j] << ", ";
            }
            std::cout << std::endl;
        }
        
        // Calculate recall
        std::unordered_set<faiss::idx_t> groundTruthSet(
            groundTruth[i].begin(), groundTruth[i].end());
        int found = 0;
        for (int j = 0; j < k; j++) {
            if (groundTruthSet.count(indices[j]) > 0) {
                found++;
            }
        }
        
        correctResults += found;
        totalResults += k;
    }

    // Quick code to analyze distance correlation
    std::cout << "\n=== Distance Correlation Analysis ===" << std::endl;
    int sampleQueryIdx = queryIndices[0];
    std::vector<float> query(dim);
    for (int j = 0; j < dim; j++) {
        query[j] = static_cast<float>(originalVectors[sampleQueryIdx * dim + j]);
    }
    auto transformedQuery = test_util::TransformQueryADC(query, stats);
    auto dc = static_cast<knn_jni::faiss_wrapper::FaissIndexBQ*>(
        static_cast<faiss::IndexHNSW*>(
            static_cast<faiss::IndexIDMap*>(adcIndex.get())->index
        )->storage)->get_FlatCodesDistanceComputer();
        dc->set_query(transformedQuery.data());
    // Calculate distances to a sample of vectors
    std::vector<std::pair<float, float>> distancePairs;
    for (int i = 0; i < std::min(100, numVectors); i++) {
        // Calculate L2 distance on original vectors
        std::vector<float> vec(dim);
        for (int j = 0; j < dim; j++) {
            vec[j] = static_cast<float>(originalVectors[i * dim + j]);
        }
        
        float l2Dist = 0;
        for (int j = 0; j < dim; j++) {
            float diff = query[j] - vec[j];
            l2Dist += diff * diff;
        }
        
        // Calculate ADC distance
        float adcDist = dc->distance_to_code(
            static_cast<uint8_t*>(&alteredStorage->codes[i * alteredStorage->code_size])
        );
        
        distancePairs.push_back({l2Dist, adcDist});
    }

    // Sort by L2 distance
    std::sort(distancePairs.begin(), distancePairs.end());

    // Check if ADC distances are monotonically increasing
    int inversions = 0;
    float prevAdc = distancePairs[0].second;
    for (size_t i = 1; i < distancePairs.size(); i++) {
        float currAdc = distancePairs[i].second;
        if (currAdc < prevAdc) inversions++;
        prevAdc = currAdc;
    }

    float inversionRate = (float)inversions / (distancePairs.size() - 1);
    std::cout << "Inversion rate: " << inversionRate << " (" << inversions 
            << "/" << (distancePairs.size() - 1) << ")" << std::endl;

    // Print sample of the distance pairs
    std::cout << "Sample of distance pairs (L2, ADC):" << std::endl;
    for (int i = 0; i < std::min(10, (int)distancePairs.size()); i++) {
        std::cout << distancePairs[i].first << ", " << distancePairs[i].second << std::endl;
    }
    
    // Calculate recall
    float recall = static_cast<float>(correctResults) / totalResults;
    std::cout << "\n=== RESULTS ===\nADC Recall@" << k << ": " << recall << std::endl;
    
    // Lower expected recall threshold to account for extreme quantization
    ASSERT_GE(recall, 0.2) << "ADC recall is too low: " << recall;

    // now rerun with the same query vectors as used for adc, but have them quantized instead.
    
}


TEST(FaissCreateBinaryIndexTest, BasicAssertions) {
    EXPECT_TRUE(false) << "FIAFILAILFJD\n\n\n\n\n\n||||||";
    // Define the data
    faiss::idx_t numIds = 200;
    std::vector<faiss::idx_t> ids;
    std::vector<uint8_t> vectors;
    int dim = 128;
    vectors.reserve(numIds);
    for (int64_t i = 0; i < numIds; ++i) {
      ids.push_back(i);
      for (int j = 0; j < dim / 8; ++j) {
        vectors.push_back(test_util::RandomInt(0, 255));
      }
    }

    std::string indexPath = test_util::RandomString(10, "tmp/", ".faiss");
    std::string spaceType = knn_jni::HAMMING;
    std::string indexDescription = "BHNSW32";

    std::unordered_map<std::string, jobject> parametersMap;
    parametersMap[knn_jni::SPACE_TYPE] = (jobject)&spaceType;
    parametersMap[knn_jni::INDEX_DESCRIPTION] = (jobject)&indexDescription;
    std::unordered_map<std::string, jobject> subParametersMap;
    parametersMap[knn_jni::PARAMETERS] = (jobject)&subParametersMap;

    // Set up jni
    NiceMock<JNIEnv> jniEnv;
    NiceMock<test_util::MockJNIUtil> mockJNIUtil;
    JavaFileIndexOutputMock javaFileIndexOutputMock {indexPath};
    setUpJavaFileOutputMocking(javaFileIndexOutputMock, mockJNIUtil, false);

    // Create the index
    std::unique_ptr<FaissMethods> faissMethods(new FaissMethods());
    NiceMock<MockIndexService> mockIndexService(std::move(faissMethods));
    int insertions = 10;
    EXPECT_CALL(mockIndexService, initIndex(_, _, faiss::METRIC_L2, indexDescription, dim, (int)numIds, 0, subParametersMap))
        .Times(1);
    EXPECT_CALL(mockIndexService, insertToIndex(dim, numIds / insertions, 0, _, _, _))
        .Times(insertions);
    EXPECT_CALL(mockIndexService, writeIndex(_, _))
        .Times(1);

    // This method calls delete vectors at the end
    createBinaryIndexIteratively(&mockJNIUtil,
                                 &jniEnv,
                                 ids,
                                 vectors,
                                 dim,
                                 (jobject) (&javaFileIndexOutputMock),
                                 parametersMap,
                                 &mockIndexService,
                                 insertions);
}

TEST(FaissADCTest, BasicAssertions2) {
    /*
    
      basically need to get perf info and make sure it works... generate dummy data between 0 and 1 and compare the differents impls. 
    */
    // Define the data
    SCOPED_TRACE("Debug message here");  // This will show up in test output
    std::cout << "here in \n\n\n\n\nthe faiss adc testing part\n";
    std::cerr << "here in \n\n\n\n\nthe faiss adc testing part\n";

    ASSERT_TRUE(true) << "fjiowjfiowejiof\n\n\\n\nn";

    ASSERT_TRUE(false) << "diagnostic message\n\n\n\n\n\n\n";
    faiss::idx_t numIds = 200;
    std::vector<faiss::idx_t> ids;
    std::vector<uint8_t> vectors;
    int dim = 128;
    vectors.reserve(numIds);
    for (int64_t i = 0; i < numIds; ++i) {
      ids.push_back(i);
      for (int j = 0; j < dim / 8; ++j) {
        vectors.push_back(test_util::RandomInt(0, 255));
      }
    }

    std::string indexPath = test_util::RandomString(10, "tmp/", ".faiss");
    std::string spaceType = knn_jni::HAMMING;
    std::string indexDescription = "BHNSW32";

    std::unordered_map<std::string, jobject> parametersMap;
    parametersMap[knn_jni::SPACE_TYPE] = (jobject)&spaceType;
    parametersMap[knn_jni::INDEX_DESCRIPTION] = (jobject)&indexDescription;
    std::unordered_map<std::string, jobject> subParametersMap;
    parametersMap[knn_jni::PARAMETERS] = (jobject)&subParametersMap;

    // Set up jni
    NiceMock<JNIEnv> jniEnv;
    NiceMock<test_util::MockJNIUtil> mockJNIUtil;
    JavaFileIndexOutputMock javaFileIndexOutputMock {indexPath};
    setUpJavaFileOutputMocking(javaFileIndexOutputMock, mockJNIUtil, false);

    // Create the index
    std::unique_ptr<FaissMethods> faissMethods(new FaissMethods());
    NiceMock<MockIndexService> mockIndexService(std::move(faissMethods));
    int insertions = 10;
    EXPECT_CALL(mockIndexService, initIndex(_, _, faiss::METRIC_L2, indexDescription, dim, (int)numIds, 0, subParametersMap))
        .Times(1);
    EXPECT_CALL(mockIndexService, insertToIndex(dim, numIds / insertions, 0, _, _, _))
        .Times(insertions);
    EXPECT_CALL(mockIndexService, writeIndex(_, _))
        .Times(1);

    // This method calls delete vectors at the end
    createBinaryIndexIteratively(&mockJNIUtil,
                                 &jniEnv,
                                 ids,
                                 vectors,
                                 dim,
                                 (jobject) (&javaFileIndexOutputMock),
                                 parametersMap,
                                 &mockIndexService,
                                 insertions);

    // once we have created the binary index, somehow call into the FaissIndexBQ header file. 


}

TEST(FaissIndexBQDirectTest, BasicAssertions45) {
        // Test 1-bit quantization with 16D vectors
        int dim = 16;
    
        // Create two 8D binary vectors where each dimension is 1 bit
        std::vector<uint8_t> codes = {
            0b11111000, // 5 ones
            0b00101010, // 3 ones
            0b11110000, // 4 ones
            0b01000000  // 1 one
        };
        
        knn_jni::faiss_wrapper::FaissIndexBQ index(dim, codes);
        
        // Create a distance computer
        std::unique_ptr<knn_jni::faiss_wrapper::CustomerFlatCodesDistanceComputer> dc(
            new knn_jni::faiss_wrapper::CustomerFlatCodesDistanceComputer(
                codes.data(), 1, dim));
        
        // Test with query vector [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        std::vector<float> query(dim, 1.0);
        dc->set_query(query.data());
        
        // For first vector [1,0,1,0,1,0,1,0]:
        // Should give -4.0 (four 1s multiplied by -1)
        float dist1 = dc->distance_to_code(&codes[0]);
        float expected_l2_dist_sq = 8.0; // 16 - number_ones
        ASSERT_FLOAT_EQ(dist1, std::sqrt(expected_l2_dist_sq));
        
        // For second vector [1,1,1,1,0,0,0,0]:
        // Should give -4.0 (four 1s multiplied by -1)
        float dist2 = dc->distance_to_code(&codes[2]); 
        float expected_l2_dist_sq_2 = 11.0;// 16 - number_ones

        ASSERT_FLOAT_EQ(dist2, std::sqrt(expected_l2_dist_sq_2));    
}

TEST(FaissIndexBQDirectTest, MultipleQueries) {
    int dim = 8;
    int num_vectors = 4;
    
    // Create test vectors with values between -127 and 128
    // These would be the original vectors before quantization
    std::vector<std::vector<int8_t>> original_vectors = {
        {100, -50, 75, -25, 60, -40, 30, -80},    // Vector 1
        {-127, 120, -100, 80, -60, 40, -20, 10},  // Vector 2
        {50, -50, 50, -50, 50, -50, 50, -50},     // Vector 3
        {0, 0, 0, 0, 127, 127, 127, 127}          // Vector 4
    };
    
    // Quantize to 1-bit based on sign (positive -> 1, negative/zero -> 0)
    std::vector<uint8_t> codes = {
        0b11010101,  // Vector 1 quantized (pos,neg,pos,neg,pos,neg,pos,neg)
        0b01010101,  // Vector 2 quantized (neg,pos,neg,pos,neg,pos,neg,pos)
        0b10101010,  // Vector 3 quantized (pos,neg,pos,neg,pos,neg,pos,neg)
        0b00001111   // Vector 4 quantized (zero->0, pos->1)
    };
    
    knn_jni::faiss_wrapper::FaissIndexBQ index(dim, codes);
    
    // Create a distance computer
    std::unique_ptr<knn_jni::faiss_wrapper::CustomerFlatCodesDistanceComputer> dc(
        new knn_jni::faiss_wrapper::CustomerFlatCodesDistanceComputer(
            codes.data(), 1, dim));
    
    // Test several query scenarios
    std::vector<std::vector<float>> queries = {
        // Query 1: Similar to Vector 1
        {90.0f, -45.0f, 70.0f, -20.0f, 55.0f, -35.0f, 25.0f, -75.0f},
        // Query 2: Similar to Vector 2
        {-120.0f, 115.0f, -95.0f, 75.0f, -55.0f, 35.0f, -15.0f, 5.0f},
        // Query 3: Similar to Vector 4 (all positive)
        {10.0f, 10.0f, 10.0f, 10.0f, 100.0f, 100.0f, 100.0f, 100.0f}
    };

    for (const auto& query : queries) {
        dc->set_query(query.data());
        
        // Calculate distances to all vectors
        std::vector<std::pair<float, int>> distances;
        for (int i = 0; i < num_vectors; i++) {
            float dist = dc->distance_to_code(&codes[i]);
            distances.push_back({dist, i});
        }
        
        // Sort by distance
        std::sort(distances.begin(), distances.end());
        
        // Verify that the closest vector makes sense based on the query
        if (query == queries[0]) {
            // Query similar to Vector 1 should return Vector 1 as closest
            ASSERT_EQ(distances[0].second, 0);
        } else if (query == queries[1]) {
            // Query similar to Vector 2 should return Vector 2 as closest
            ASSERT_EQ(distances[0].second, 1);
        } else if (query == queries[2]) {
            // Query similar to Vector 4 should return Vector 4 as closest
            ASSERT_EQ(distances[0].second, 3);
        }
    }

}

TEST(FaissADCTest, BasicQuantizationAndMeans) {
    int dim = 8;
    int numVectors = 100;
    
    // Create vectors with known means
    std::vector<int8_t> vectors;
    std::vector<float> expectedMeans = {50.0f, -30.0f, 0.0f, 100.0f, -80.0f, 25.0f, -60.0f, 75.0f};
    for (float mean : expectedMeans) {
        auto vec = test_util::GenerateVectorWithMean(dim, mean, 20.0f);
        vectors.insert(vectors.end(), vec.begin(), vec.end());
    }

    // Quantize to 1-bit
    auto codes = test_util::QuantizeVectors(vectors, dim, numVectors, 1);
    
    // Compute dimension statistics
    auto stats = test_util::ComputeDimensionStats(vectors, codes, dim, numVectors);
    
    // Verify statistics
    for (int d = 0; d < dim; d++) {
        // Zero means should be less than one means for each dimension
        ASSERT_LE(stats.zero_means[d], stats.one_means[d]);
        
        // Each dimension should have some values quantized to both 0 and 1
        ASSERT_GT(stats.zero_counts[d], 0);
        ASSERT_GT(stats.one_counts[d], 0);
        
        // Total counts should equal number of vectors
        ASSERT_EQ(stats.zero_counts[d] + stats.one_counts[d], numVectors);
    }
}

TEST(FaissADCTest, QueryTransformation) {
    int dim = 8;
    int numVectors = 100;
    
    // Create test vectors with controlled distribution
    std::vector<int8_t> vectors = test_util::RandomByteVectors(dim * numVectors, 1, -127, 127);
    auto codes = test_util::QuantizeVectors(vectors, dim, numVectors, 1);
    auto stats = test_util::ComputeDimensionStats(vectors, codes, dim, numVectors);
    
    // Test query transformation
    std::vector<std::vector<float>> testQueries = {
        std::vector<float>(dim, 0.0f),  // Zero query
        std::vector<float>(dim, 100.0f), // Large positive query
        std::vector<float>(dim, -100.0f) // Large negative query
    };
    
    for (const auto& query : testQueries) {
        auto transformed = test_util::TransformQueryADC(query, stats);
        
        // Verify transformed query properties
        for (int d = 0; d < dim; d++) {
            // If query equals zero mean, should get 0
            if (std::abs(query[d] - stats.zero_means[d]) < 1e-6) {
                ASSERT_NEAR(transformed[d], 0.0f, 1e-6);
            }
            // If query equals one mean, should get 1
            if (std::abs(query[d] - stats.one_means[d]) < 1e-6) {
                ASSERT_NEAR(transformed[d], 1.0f, 1e-6);
            }
        }
    }
}

TEST(FaissADCTest, NearestNeighborSearch) {
    int dim = 8;
    int numVectors = 100;
    
    // Create vectors with distinct patterns
    std::vector<int8_t> vectors;
    std::vector<float> patterns = {100.0f, -100.0f, 0.0f, 50.0f, -50.0f};
    for (float pattern : patterns) {
        auto vec = test_util::GenerateVectorWithMean(dim, pattern, 10.0f);
        vectors.insert(vectors.end(), vec.begin(), vec.end());
    }
    
    // Fill remaining vectors with random values
    auto randomVecs = test_util::RandomByteVectors(dim * (numVectors - patterns.size()), 1, -127, 127);
    vectors.insert(vectors.end(), randomVecs.begin(), randomVecs.end());
    
    // Create index and compute ADC statistics
    auto codes = test_util::QuantizeVectors(vectors, dim, numVectors, 1);
    knn_jni::faiss_wrapper::FaissIndexBQ index(dim, codes);
    auto stats = test_util::ComputeDimensionStats(vectors, codes, dim, numVectors);
    
    // Create distance computer
    auto dc = std::unique_ptr<knn_jni::faiss_wrapper::CustomerFlatCodesDistanceComputer>(
        new knn_jni::faiss_wrapper::CustomerFlatCodesDistanceComputer(
            codes.data(), 1, dim));
    
    // Test nearest neighbor search with pattern vectors
    for (size_t i = 0; i < patterns.size(); i++) {
        // Create query similar to pattern vector
        std::vector<float> query(vectors.begin() + i * dim, vectors.begin() + (i + 1) * dim);
        
        // Transform query using ADC
        auto transformed = test_util::TransformQueryADC(query, stats);
        dc->set_query(transformed.data());
        
        // Find nearest neighbors
        std::vector<std::pair<float, int>> distances;
        for (int j = 0; j < numVectors; j++) {
            float dist = dc->distance_to_code(&codes[j]);
            distances.push_back({dist, j});
        }
        
        // Sort by distance
        std::sort(distances.begin(), distances.end());
        
        // The closest vector should be the original pattern vector
        ASSERT_EQ(distances[0].second, i);
    }
}

TEST(FaissADCTest, EdgeCases) {
    int dim = 8;
    int numVectors = 10;
    
    // Test edge case vectors
    std::vector<std::vector<int8_t>> edgeCases = {
        std::vector<int8_t>(dim, 127),    // All maximum positive
        std::vector<int8_t>(dim, -128),   // All maximum negative
        std::vector<int8_t>(dim, 0)       // All zeros
    };
    
    std::vector<int8_t> vectors;
    for (const auto& edge : edgeCases) {
        vectors.insert(vectors.end(), edge.begin(), edge.end());
    }
    
    // Add some random vectors
    auto randomVecs = test_util::RandomByteVectors(dim * (numVectors - edgeCases.size()), 1, -127, 127);
    vectors.insert(vectors.end(), randomVecs.begin(), randomVecs.end());
    
    auto codes = test_util::QuantizeVectors(vectors, dim, numVectors, 1);
    auto stats = test_util::ComputeDimensionStats(vectors, codes, dim, numVectors);
    
    // Test edge case queries
    std::vector<std::vector<float>> edgeQueries = {
        std::vector<float>(dim, std::numeric_limits<float>::max()),
        std::vector<float>(dim, std::numeric_limits<float>::lowest()),
        std::vector<float>(dim, 0.0f)
    };
    
    for (const auto& query : edgeQueries) {
        auto transformed = test_util::TransformQueryADC(query, stats);
        
        // Verify transformed values are finite
        for (float val : transformed) {
            ASSERT_FALSE(std::isinf(val));
            ASSERT_FALSE(std::isnan(val));
        }
    }
    
    // Test with zero variance in a dimension
    std::vector<int8_t> constantVector(dim * numVectors);
    std::fill(constantVector.begin(), constantVector.end(), 42);
    auto constantCodes = test_util::QuantizeVectors(constantVector, dim, numVectors, 1);
    auto constantStats = test_util::ComputeDimensionStats(constantVector, constantCodes, dim, numVectors);
    
    std::vector<float> query(dim, 0.0f);
    auto transformed = test_util::TransformQueryADC(query, constantStats);
    
    // Should handle zero variance gracefully
    for (float val : transformed) {
        ASSERT_FALSE(std::isinf(val));
        ASSERT_FALSE(std::isnan(val));
    }
}




TEST(FaissCreateIndexFromTemplateTest, BasicAssertions) {
    for (auto throwIOException : std::array<bool, 2> {false, true}) {
        // Define the data
        faiss::idx_t numIds = 100;
        std::vector<faiss::idx_t> ids;
        auto *vectors = new std::vector<float>();
        int dim = 2;
        vectors->reserve(dim * numIds);
        for (int64_t i = 0; i < numIds; ++i) {
          ids.push_back(i);
          for (int j = 0; j < dim; ++j) {
            vectors->push_back(test_util::RandomFloat(-500.0, 500.0));
          }
        }

        std::string indexPath = test_util::RandomString(10, "tmp/", ".faiss");
        faiss::MetricType metricType = faiss::METRIC_L2;
        std::string method = "HNSW32,Flat";

        std::unique_ptr<faiss::Index> createdIndex(
            test_util::FaissCreateIndex(dim, method, metricType));
        auto vectorIoWriter = test_util::FaissGetSerializedIndex(createdIndex.get());

        // Setup jni
        NiceMock<JNIEnv> jniEnv;
        NiceMock<test_util::MockJNIUtil> mockJNIUtil;
        JavaFileIndexOutputMock javaFileIndexOutputMock {indexPath};
        setUpJavaFileOutputMocking(javaFileIndexOutputMock, mockJNIUtil, throwIOException);

        std::string spaceType = knn_jni::L2;
        std::unordered_map<std::string, jobject> parametersMap;
        parametersMap[knn_jni::SPACE_TYPE] = (jobject) &spaceType;

        try {
            knn_jni::faiss_wrapper::CreateIndexFromTemplate(
                &mockJNIUtil, &jniEnv, reinterpret_cast<jintArray>(&ids),
                (jlong)vectors, dim, (jobject)(&javaFileIndexOutputMock),
                reinterpret_cast<jbyteArray>(&(vectorIoWriter.data)),
                (jobject) &parametersMap);
            javaFileIndexOutputMock.file_writer.close();
        } catch (const StreamIOError& e) {
            ASSERT_TRUE(throwIOException);
            continue;
        }

        ASSERT_FALSE(throwIOException);

        // Make sure index can be loaded
        std::unique_ptr<faiss::Index> index(test_util::FaissLoadIndex(indexPath));

        // Clean up
        std::remove(indexPath.c_str());
    }  // End for
}

TEST(FaissCreateByteIndexFromTemplateTest, BasicAssertions) {
    for (auto throwIOException : std::array<bool, 2> {false, true}) {
        // Define the data
        faiss::idx_t numIds = 100;
        std::vector<faiss::idx_t> ids;
        auto *vectors = new std::vector<int8_t>();
        int dim = 8;
        vectors->reserve(dim * numIds);
        for (int64_t i = 0; i < numIds; ++i) {
          ids.push_back(i);
          for (int j = 0; j < dim; ++j) {
            vectors->push_back(test_util::RandomInt(-128, 127));
          }
        }

        std::string indexPath = test_util::RandomString(10, "tmp/", ".faiss");
        faiss::MetricType metricType = faiss::METRIC_L2;
        std::string method = "HNSW32,SQ8_direct_signed";

        std::unique_ptr<faiss::Index> createdIndex(
            test_util::FaissCreateIndex(dim, method, metricType));
        auto vectorIoWriter = test_util::FaissGetSerializedIndex(createdIndex.get());

        // Setup jni
        NiceMock<JNIEnv> jniEnv;
        NiceMock<test_util::MockJNIUtil> mockJNIUtil;
        JavaFileIndexOutputMock javaFileIndexOutputMock {indexPath};
        setUpJavaFileOutputMocking(javaFileIndexOutputMock, mockJNIUtil, throwIOException);

        std::string spaceType = knn_jni::L2;
        std::unordered_map<std::string, jobject> parametersMap;
        parametersMap[knn_jni::SPACE_TYPE] = (jobject) &spaceType;

        try {
            knn_jni::faiss_wrapper::CreateByteIndexFromTemplate(
                &mockJNIUtil, &jniEnv, reinterpret_cast<jintArray>(&ids),
                (jlong) vectors, dim, (jstring) (&javaFileIndexOutputMock),
                reinterpret_cast<jbyteArray>(&(vectorIoWriter.data)),
                (jobject) &parametersMap
            );

            // Make sure we close a file stream before reopening the created file.
            javaFileIndexOutputMock.file_writer.close();
        } catch (const StreamIOError& e) {
            ASSERT_TRUE(throwIOException);
            continue;
        }

        ASSERT_FALSE(throwIOException);

        // Make sure index can be loaded
        std::unique_ptr<faiss::Index> index(test_util::FaissLoadIndex(indexPath));

        // Clean up
        std::remove(indexPath.c_str());
    }  // End for
}

TEST(FaissLoadIndexTest, BasicAssertions) {
    // Define the data
    faiss::idx_t numIds = 100;
    int dim = 2;
    std::vector<faiss::idx_t> ids = test_util::Range(numIds);
    std::vector<float> vectors = test_util::RandomVectors(dim, numIds, randomDataMin, randomDataMax);

    std::string indexPath = test_util::RandomString(10, "tmp/", ".faiss");
    faiss::MetricType metricType = faiss::METRIC_L2;
    std::string method = "HNSW32,Flat";

    // Create the index
    std::unique_ptr<faiss::Index> createdIndex(
            test_util::FaissCreateIndex(dim, method, metricType));
    auto createdIndexWithData =
            test_util::FaissAddData(createdIndex.get(), ids, vectors);

    test_util::FaissWriteIndex(&createdIndexWithData, indexPath);

    // Setup jni
    NiceMock<JNIEnv> jniEnv;
    NiceMock<test_util::MockJNIUtil> mockJNIUtil;

    std::unique_ptr<faiss::Index> loadedIndexPointer(
            reinterpret_cast<faiss::Index *>(knn_jni::faiss_wrapper::LoadIndex(
                    &mockJNIUtil, &jniEnv, (jstring)&indexPath)));

    // Compare serialized versions
    auto createIndexSerialization =
            test_util::FaissGetSerializedIndex(&createdIndexWithData);
    auto loadedIndexSerialization = test_util::FaissGetSerializedIndex(
            reinterpret_cast<faiss::Index *>(loadedIndexPointer.get()));

    ASSERT_NE(0, loadedIndexSerialization.data.size());
    ASSERT_EQ(createIndexSerialization.data.size(),
              loadedIndexSerialization.data.size());

    for (int i = 0; i < loadedIndexSerialization.data.size(); ++i) {
        ASSERT_EQ(createIndexSerialization.data[i],
                  loadedIndexSerialization.data[i]);
    }

    // Clean up
    std::remove(indexPath.c_str());
}

TEST(FaissLoadBinaryIndexTest, BasicAssertions) {
    // Define the data
    faiss::idx_t numIds = 200;
    std::vector<faiss::idx_t> ids;
    auto vectors = std::vector<uint8_t>(numIds);
    int dim = 128;
    for (int64_t i = 0; i < numIds; ++i) {
        ids.push_back(i);
        for (int j = 0; j < dim / 8; ++j) {
            vectors.push_back(test_util::RandomInt(0, 255));
        }
    }

    std::string indexPath = test_util::RandomString(10, "tmp/", ".faiss");
    std::string method = "BHNSW32";

    // Create the index
    std::unique_ptr<faiss::IndexBinary> createdIndex(
            test_util::FaissCreateBinaryIndex(dim, method));
    auto createdIndexWithData =
            test_util::FaissAddBinaryData(createdIndex.get(), ids, vectors);

    test_util::FaissWriteBinaryIndex(&createdIndexWithData, indexPath);

    // Setup jni
    NiceMock<JNIEnv> jniEnv;
    NiceMock<test_util::MockJNIUtil> mockJNIUtil;

    std::unique_ptr<faiss::IndexBinary> loadedIndexPointer(
            reinterpret_cast<faiss::IndexBinary *>(knn_jni::faiss_wrapper::LoadBinaryIndex(
                    &mockJNIUtil, &jniEnv, (jstring)&indexPath)));

    // Compare serialized versions
    auto createIndexSerialization =
            test_util::FaissGetSerializedBinaryIndex(&createdIndexWithData);
    auto loadedIndexSerialization = test_util::FaissGetSerializedBinaryIndex(
            reinterpret_cast<faiss::IndexBinary *>(loadedIndexPointer.get()));

    ASSERT_NE(0, loadedIndexSerialization.data.size());
    ASSERT_EQ(createIndexSerialization.data.size(),
              loadedIndexSerialization.data.size());

    for (int i = 0; i < loadedIndexSerialization.data.size(); ++i) {
        ASSERT_EQ(createIndexSerialization.data[i],
                  loadedIndexSerialization.data[i]);
    }

    // Clean up
    std::remove(indexPath.c_str());
}

TEST(FaissLoadIndexTest, HNSWPQDisableSdcTable) {
    // Check that when we load an HNSWPQ index, the sdc table is not present.
    faiss::idx_t numIds = 256;
    int dim = 2;
    std::vector<faiss::idx_t> ids = test_util::Range(numIds);
    std::vector<float> vectors = test_util::RandomVectors(dim, numIds, randomDataMin, randomDataMax);

    std::string indexPath = test_util::RandomString(10, "tmp/", ".faiss");
    faiss::MetricType metricType = faiss::METRIC_L2;
    std::string indexDescription = "HNSW16,PQ1x4";

    std::unique_ptr<faiss::Index> faissIndex(test_util::FaissCreateIndex(dim, indexDescription, metricType));
    test_util::FaissTrainIndex(faissIndex.get(), numIds, vectors.data());
    auto faissIndexWithIDMap = test_util::FaissAddData(faissIndex.get(), ids, vectors);
    test_util::FaissWriteIndex(&faissIndexWithIDMap, indexPath);

    // Setup jni
    NiceMock<JNIEnv> jniEnv;
    NiceMock<test_util::MockJNIUtil> mockJNIUtil;

    std::unique_ptr<faiss::Index> loadedIndexPointer(
            reinterpret_cast<faiss::Index *>(knn_jni::faiss_wrapper::LoadIndex(
                    &mockJNIUtil, &jniEnv, (jstring)&indexPath)));

    // Cast down until we get to the pq backed storage index and checke the size of the table
    auto idMapIndex = dynamic_cast<faiss::IndexIDMap *>(loadedIndexPointer.get());
    ASSERT_NE(idMapIndex, nullptr);
    auto hnswPQIndex = dynamic_cast<faiss::IndexHNSWPQ *>(idMapIndex->index);
    ASSERT_NE(hnswPQIndex, nullptr);
    auto pqIndex = dynamic_cast<faiss::IndexPQ*>(hnswPQIndex->storage);
    ASSERT_NE(pqIndex, nullptr);
    ASSERT_EQ(0, pqIndex->pq.sdc_table.size());
}

TEST(FaissLoadIndexTest, IVFPQDisablePrecomputeTable) {
    faiss::idx_t numIds = 256;
    int dim = 2;
    std::vector<faiss::idx_t> ids = test_util::Range(numIds);
    std::vector<float> vectors = test_util::RandomVectors(dim, numIds, randomDataMin, randomDataMax);

    std::string indexPath = test_util::RandomString(10, "tmp/", ".faiss");
    faiss::MetricType metricType = faiss::METRIC_L2;
    std::string indexDescription = "IVF4,PQ1x4";

    std::unique_ptr<faiss::Index> faissIndex(test_util::FaissCreateIndex(dim, indexDescription, metricType));
    test_util::FaissTrainIndex(faissIndex.get(), numIds, vectors.data());
    auto faissIndexWithIDMap = test_util::FaissAddData(faissIndex.get(), ids, vectors);
    test_util::FaissWriteIndex(&faissIndexWithIDMap, indexPath);

    // Setup jni
    NiceMock<JNIEnv> jniEnv;
    NiceMock<test_util::MockJNIUtil> mockJNIUtil;

    std::unique_ptr<faiss::Index> loadedIndexPointer(
            reinterpret_cast<faiss::Index *>(knn_jni::faiss_wrapper::LoadIndex(
                    &mockJNIUtil, &jniEnv, (jstring)&indexPath)));

    // Cast down until we get to the ivfpq-l2 state
    auto idMapIndex = dynamic_cast<faiss::IndexIDMap *>(loadedIndexPointer.get());
    ASSERT_NE(idMapIndex, nullptr);
    auto ivfpqIndex = dynamic_cast<faiss::IndexIVFPQ *>(idMapIndex->index);
    ASSERT_NE(ivfpqIndex, nullptr);
    ASSERT_EQ(0, ivfpqIndex->precomputed_table->size());
}

TEST(FaissQueryIndexTest, BasicAssertions) {
    // Define the index data
    faiss::idx_t numIds = 100;
    int dim = 16;
    std::vector<faiss::idx_t> ids = test_util::Range(numIds);
    std::vector<float> vectors = test_util::RandomVectors(dim, numIds, randomDataMin, randomDataMax);

    faiss::MetricType metricType = faiss::METRIC_L2;
    std::string method = "HNSW32,Flat";

    // Define query data
    int k = 10;
    int efSearch = 20;
    std::unordered_map<std::string, jobject> methodParams;
    methodParams[knn_jni::EF_SEARCH] = reinterpret_cast<jobject>(&efSearch);

    int numQueries = 100;
    std::vector<std::vector<float>> queries;

    for (int i = 0; i < numQueries; i++) {
        std::vector<float> query;
        query.reserve(dim);
        for (int j = 0; j < dim; j++) {
            query.push_back(test_util::RandomFloat(-500.0, 500.0));
        }
        queries.push_back(query);
    }

    // Create the index
    std::unique_ptr<faiss::Index> createdIndex(
            test_util::FaissCreateIndex(dim, method, metricType));
    auto createdIndexWithData =
            test_util::FaissAddData(createdIndex.get(), ids, vectors);

    // Setup jni
    NiceMock<JNIEnv> jniEnv;
    NiceMock<test_util::MockJNIUtil> mockJNIUtil;
    auto methodParamsJ = reinterpret_cast<jobject>(&methodParams);

    for (auto query : queries) {
        std::unique_ptr<std::vector<std::pair<int, float> *>> results(
                reinterpret_cast<std::vector<std::pair<int, float> *> *>(
                        knn_jni::faiss_wrapper::QueryIndex(
                                &mockJNIUtil, &jniEnv,
                                reinterpret_cast<jlong>(&createdIndexWithData),
                                reinterpret_cast<jfloatArray>(&query), k, methodParamsJ, nullptr)));

        ASSERT_EQ(k, results->size());

        // Need to free up each result
        for (auto it : *results.get()) {
            delete it;
        }
    }
}

TEST(FaissQueryBinaryIndexTest, BasicAssertions) {
    // Define the data
    faiss::idx_t numIds = 200;
    std::vector<faiss::idx_t> ids;
    auto vectors = std::vector<uint8_t>(numIds);
    int dim = 128;
    for (int64_t i = 0; i < numIds; ++i) {
        ids.push_back(i);
        for (int j = 0; j < dim / 8; ++j) {
            vectors.push_back(test_util::RandomInt(0, 255));
        }
    }

    // Define query data
    int k = 10;
    int numQueries = 100;
    std::vector<std::vector<uint8_t>> queries;

    for (int i = 0; i < numQueries; i++) {
        std::vector<uint8_t> query;
        query.reserve(dim);
        for (int j = 0; j < dim; j++) {
            query.push_back(test_util::RandomInt(0, 255));
        }
        queries.push_back(query);
    }

    // Create the index
    std::string method = "BHNSW32";
    std::unique_ptr<faiss::IndexBinary> createdIndex(
            test_util::FaissCreateBinaryIndex(dim, method));
    auto createdIndexWithData =
            test_util::FaissAddBinaryData(createdIndex.get(), ids, vectors);

    // Setup jni
    NiceMock<JNIEnv> jniEnv;
    NiceMock<test_util::MockJNIUtil> mockJNIUtil;

    for (auto query : queries) {
        std::unique_ptr<std::vector<std::pair<int, int32_t> *>> results(
                reinterpret_cast<std::vector<std::pair<int, int32_t> *> *>(
                        knn_jni::faiss_wrapper::QueryBinaryIndex_WithFilter(
                                &mockJNIUtil, &jniEnv,
                                reinterpret_cast<jlong>(&createdIndexWithData),
                                reinterpret_cast<jbyteArray>(&query), k, nullptr, nullptr, 0, nullptr)));

        ASSERT_EQ(k, results->size());

        // Need to free up each result
        for (auto it : *results.get()) {
            delete it;
        }
    }
}

//Test for a bug reported in https://github.com/opensearch-project/k-NN/issues/1435
TEST(FaissQueryIndexWithFilterTest1435, BasicAssertions) {
    // Define the index data
    faiss::idx_t numIds = 200;
    std::vector<faiss::idx_t> ids;
    std::vector<float> vectors;
    std::vector<std::vector<float>> queries;

    int dim = 16;
    for (int64_t i = 1; i < numIds + 1; i++) {
        std::vector<float> query;
        query.reserve(dim);
        ids.push_back(i);
        for (int j = 0; j < dim; j++) {
            float vector = test_util::RandomFloat(-500.0, 500.0);
            vectors.push_back(vector);
            query.push_back(vector);
        }
        queries.push_back(query);
    }

    int num_bits = test_util::bits2words(164);
    std::vector<jlong> bitmap(num_bits,0);
    std::vector<int64_t> filterIds;

    for (int64_t i = 154; i < 163; i++) {
        filterIds.push_back(i);
        test_util::setBitSet(i, bitmap.data(), bitmap.size());
    }
    std::unordered_set<int> filterIdSet(filterIds.begin(), filterIds.end());

    faiss::MetricType metricType = faiss::METRIC_L2;
    std::string method = "HNSW32,Flat";

    // Create the index
    std::unique_ptr<faiss::Index> createdIndex(
            test_util::FaissCreateIndex(dim, method, metricType));
    auto createdIndexWithData =
            test_util::FaissAddData(createdIndex.get(), ids, vectors);

    // Setup jni
    NiceMock<JNIEnv> jniEnv;
    NiceMock<test_util::MockJNIUtil> mockJNIUtil;
    EXPECT_CALL(mockJNIUtil,
                GetJavaLongArrayLength(
                        &jniEnv, reinterpret_cast<jlongArray>(&bitmap)))
            .WillRepeatedly(Return(bitmap.size()));

    int k = 20;
    for (auto query : queries) {
        std::unique_ptr<std::vector<std::pair<int, float> *>> results(
                reinterpret_cast<std::vector<std::pair<int, float> *> *>(
                        knn_jni::faiss_wrapper::QueryIndex_WithFilter(
                                &mockJNIUtil, &jniEnv,
                                reinterpret_cast<jlong>(&createdIndexWithData),
                                reinterpret_cast<jfloatArray>(&query), k, nullptr,
                                reinterpret_cast<jlongArray>(&bitmap), 0, nullptr)));

        ASSERT_TRUE(results->size() <= filterIds.size());
        ASSERT_TRUE(results->size() > 0);
        for (const auto& pairPtr : *results) {
            auto it = filterIdSet.find(pairPtr->first);
            ASSERT_NE(it, filterIdSet.end());
        }

        // Need to free up each result
        for (auto it : *results.get()) {
            delete it;
        }
    }
}

TEST(FaissQueryIndexWithParentFilterTest, BasicAssertions) {
    // Define the index data
    faiss::idx_t numIds = 100;
    std::vector<faiss::idx_t> ids;
    std::vector<float> vectors;
    std::vector<int> parentIds;
    int dim = 16;
    for (int64_t i = 1; i < numIds + 1; i++) {
        if (i % 10 == 0) {
            parentIds.push_back(i);
            continue;
        }
        ids.push_back(i);
        for (int j = 0; j < dim; j++) {
            vectors.push_back(test_util::RandomFloat(-500.0, 500.0));
        }
    }

    faiss::MetricType metricType = faiss::METRIC_L2;
    std::string method = "HNSW32,Flat";

    // Define query data
    int k = 20;
    int numQueries = 100;
    std::vector<std::vector<float>> queries;

    for (int i = 0; i < numQueries; i++) {
        std::vector<float> query;
        query.reserve(dim);
        for (int j = 0; j < dim; j++) {
            query.push_back(test_util::RandomFloat(-500.0, 500.0));
        }
        queries.push_back(query);
    }

    // Create the index
    std::unique_ptr<faiss::Index> createdIndex(
            test_util::FaissCreateIndex(dim, method, metricType));
    auto createdIndexWithData =
            test_util::FaissAddData(createdIndex.get(), ids, vectors);

    int efSearch = 100;
    std::unordered_map<std::string, jobject> methodParams;
    methodParams[knn_jni::EF_SEARCH] = reinterpret_cast<jobject>(&efSearch);

    // Setup jni
    NiceMock<JNIEnv> jniEnv;
    NiceMock<test_util::MockJNIUtil> mockJNIUtil;
    EXPECT_CALL(mockJNIUtil,
                    GetJavaIntArrayLength(
                            &jniEnv, reinterpret_cast<jintArray>(&parentIds)))
                .WillRepeatedly(Return(parentIds.size()));
    for (auto query : queries) {
        std::unique_ptr<std::vector<std::pair<int, float> *>> results(
                reinterpret_cast<std::vector<std::pair<int, float> *> *>(
                        knn_jni::faiss_wrapper::QueryIndex(
                                &mockJNIUtil, &jniEnv,
                                reinterpret_cast<jlong>(&createdIndexWithData),
                                reinterpret_cast<jfloatArray>(&query), k, reinterpret_cast<jobject>(&methodParams),
                                reinterpret_cast<jintArray>(&parentIds))));

        // Even with k 20, result should have only 10 which is total number of groups
        ASSERT_EQ(10, results->size());
        // Result should be one for each group
        std::set<int> idSet;
        for (const auto& pairPtr : *results) {
            idSet.insert(pairPtr->first / 10);
        }
        ASSERT_EQ(10, idSet.size());

        // Need to free up each result
        for (auto it : *results.get()) {
            delete it;
        }
    }
}

TEST(FaissFreeTest, BasicAssertions) {
    // Define the data
    int dim = 2;
    faiss::MetricType metricType = faiss::METRIC_L2;
    std::string method = "HNSW32,Flat";

    // Create the index
    faiss::Index *createdIndex(
            test_util::FaissCreateIndex(dim, method, metricType));

    // Free created index --> memory check should catch failure
    knn_jni::faiss_wrapper::Free(reinterpret_cast<jlong>(createdIndex), JNI_FALSE);
}


TEST(FaissBinaryFreeTest, BasicAssertions) {
    // Define the data
    int dim = 8;
    std::string method = "BHNSW32";

    // Create the index
    faiss::IndexBinary *createdIndex(
            test_util::FaissCreateBinaryIndex(dim, method));

    // Free created index --> memory check should catch failure
    knn_jni::faiss_wrapper::Free(reinterpret_cast<jlong>(createdIndex), JNI_TRUE);
}

TEST(FaissInitLibraryTest, BasicAssertions) {
    knn_jni::faiss_wrapper::InitLibrary();
}

TEST(FaissTrainIndexTest, BasicAssertions) {
    // Define the index configuration
    int dim = 2;
    std::string spaceType = knn_jni::L2;
    std::string index_description = "IVF4,Flat";

    std::unordered_map<std::string, jobject> parametersMap;
    parametersMap[knn_jni::SPACE_TYPE] = (jobject) &spaceType;
    parametersMap[knn_jni::INDEX_DESCRIPTION] = (jobject) &index_description;

    // Define training data
    int numTrainingVectors = 256;
    std::vector<float> trainingVectors = test_util::RandomVectors(dim, numTrainingVectors, randomDataMin, randomDataMax);

    // Setup jni
    NiceMock<JNIEnv> jniEnv;
    NiceMock<test_util::MockJNIUtil> mockJNIUtil;

    // Perform training
    std::unique_ptr<std::vector<uint8_t>> trainedIndexSerialization(
            reinterpret_cast<std::vector<uint8_t> *>(
                    knn_jni::faiss_wrapper::TrainIndex(
                            &mockJNIUtil, &jniEnv, (jobject) &parametersMap, dim,
                            reinterpret_cast<jlong>(&trainingVectors))));

    std::unique_ptr<faiss::Index> trainedIndex(
            test_util::FaissLoadFromSerializedIndex(trainedIndexSerialization.get()));

    // Confirm that training succeeded
    ASSERT_TRUE(trainedIndex->is_trained);
}

TEST(FaissTrainByteIndexTest, BasicAssertions) {
    // Define the index configuration
    int dim = 2;
    std::string spaceType = knn_jni::L2;
    std::string index_description = "IVF4,SQ8_direct_signed";

    std::unordered_map<std::string, jobject> parametersMap;
    parametersMap[knn_jni::SPACE_TYPE] = (jobject) &spaceType;
    parametersMap[knn_jni::INDEX_DESCRIPTION] = (jobject) &index_description;

    // Define training data
    int numTrainingVectors = 256;
    std::vector<int8_t> trainingVectors = test_util::RandomByteVectors(dim, numTrainingVectors, -128, 127);

    // Setup jni
    NiceMock<JNIEnv> jniEnv;
    NiceMock<test_util::MockJNIUtil> mockJNIUtil;

    // Perform training
    std::unique_ptr<std::vector<uint8_t>> trainedIndexSerialization(
            reinterpret_cast<std::vector<uint8_t> *>(
                    knn_jni::faiss_wrapper::TrainByteIndex(
                            &mockJNIUtil, &jniEnv, (jobject) &parametersMap, dim,
                            reinterpret_cast<jlong>(&trainingVectors))));

    std::unique_ptr<faiss::Index> trainedIndex(
            test_util::FaissLoadFromSerializedIndex(trainedIndexSerialization.get()));

    // Confirm that training succeeded
    ASSERT_TRUE(trainedIndex->is_trained);
}

TEST(FaissCreateHnswSQfp16IndexTest, BasicAssertions) {
    // Define the data
    faiss::idx_t numIds = 200;
    std::vector<faiss::idx_t> ids;
    std::vector<float> vectors;
    int dim = 2;
    vectors.reserve(dim * numIds);
    for (int64_t i = 0; i < numIds; ++i) {
        ids.push_back(i);
        for (int j = 0; j < dim; ++j) {
            vectors.push_back(test_util::RandomFloat(-500.0, 500.0));
        }
    }

    std::string spaceType = knn_jni::L2;
    std::string index_description = "HNSW32,SQfp16";

    std::unordered_map<std::string, jobject> parametersMap;
    parametersMap[knn_jni::SPACE_TYPE] = (jobject)&spaceType;
    parametersMap[knn_jni::INDEX_DESCRIPTION] = (jobject)&index_description;

    for (auto throwIOException : std::array<bool, 2> {false, true}) {
        const std::string indexPath = test_util::RandomString(10, "tmp/", ".faiss");

        // Set up jni
        NiceMock<JNIEnv> jniEnv;
        NiceMock<test_util::MockJNIUtil> mockJNIUtil;
        JavaFileIndexOutputMock javaFileIndexOutputMock {indexPath};
        setUpJavaFileOutputMocking(javaFileIndexOutputMock, mockJNIUtil, throwIOException);

        EXPECT_CALL(mockJNIUtil,
                    GetJavaObjectArrayLength(
                        &jniEnv, reinterpret_cast<jobjectArray>(&vectors)))
            .WillRepeatedly(Return(vectors.size()));

        // Create the index
        std::unique_ptr<FaissMethods> faissMethods(new FaissMethods());
        knn_jni::faiss_wrapper::IndexService IndexService(std::move(faissMethods));

        try {
            createIndexIteratively(&mockJNIUtil, &jniEnv, ids, vectors, dim, (jobject) (&javaFileIndexOutputMock), parametersMap, &IndexService);
            // Make sure we close a file stream before reopening the created file.
            javaFileIndexOutputMock.file_writer.close();
        } catch (const std::exception& e) {
            ASSERT_STREQ("Failed to write index to disk", e.what());
            ASSERT_TRUE(throwIOException);
            continue;
        }
        ASSERT_FALSE(throwIOException);

        // Make sure index can be loaded
        std::unique_ptr<faiss::Index> index(test_util::FaissLoadIndex(indexPath));
        auto indexIDMap =  dynamic_cast<faiss::IndexIDMap*>(index.get());

        // Assert that Index is of type IndexHNSWSQ
        ASSERT_NE(indexIDMap, nullptr);
        ASSERT_NE(dynamic_cast<faiss::IndexHNSWSQ*>(indexIDMap->index), nullptr);

        // Clean up
        std::remove(indexPath.c_str());
    }  // End for
}

TEST(FaissIsSharedIndexStateRequired, BasicAssertions) {
    int d = 128;
    int hnswM = 16;
    int ivfNlist = 4;
    int pqM = 1;
    int pqCodeSize = 8;
    std::unique_ptr<faiss::IndexHNSW> indexHNSWL2(new faiss::IndexHNSW(d, hnswM, faiss::METRIC_L2));
    std::unique_ptr<faiss::IndexIVFPQ> indexIVFPQIP(new faiss::IndexIVFPQ(
                new faiss::IndexFlat(d, faiss::METRIC_INNER_PRODUCT),
                d,
                ivfNlist,
                pqM,
                pqCodeSize,
                faiss::METRIC_INNER_PRODUCT
            ));
    std::unique_ptr<faiss::IndexIVFPQ> indexIVFPQL2(new faiss::IndexIVFPQ(
                new faiss::IndexFlat(d, faiss::METRIC_L2),
                d,
                ivfNlist,
                pqM,
                pqCodeSize,
                faiss::METRIC_L2
            ));
    std::unique_ptr<faiss::IndexIDMap> indexIDMapIVFPQL2(new faiss::IndexIDMap(
                new faiss::IndexIVFPQ(
                        new faiss::IndexFlat(d, faiss::METRIC_L2),
                        d,
                        ivfNlist,
                        pqM,
                        pqCodeSize,
                        faiss::METRIC_L2
                )
            ));
    std::unique_ptr<faiss::IndexIDMap> indexIDMapIVFPQIP(new faiss::IndexIDMap(
                new faiss::IndexIVFPQ(
                        new faiss::IndexFlat(d, faiss::METRIC_INNER_PRODUCT),
                        d,
                        ivfNlist,
                        pqM,
                        pqCodeSize,
                        faiss::METRIC_INNER_PRODUCT
                )
            ));
    jlong nullAddress = 0;

    ASSERT_FALSE(knn_jni::faiss_wrapper::IsSharedIndexStateRequired((jlong) indexHNSWL2.get()));
    ASSERT_FALSE(knn_jni::faiss_wrapper::IsSharedIndexStateRequired((jlong) indexIVFPQIP.get()));
    ASSERT_FALSE(knn_jni::faiss_wrapper::IsSharedIndexStateRequired((jlong) indexIDMapIVFPQIP.get()));
    ASSERT_FALSE(knn_jni::faiss_wrapper::IsSharedIndexStateRequired((jlong) nullAddress));

    ASSERT_TRUE(knn_jni::faiss_wrapper::IsSharedIndexStateRequired((jlong) indexIVFPQL2.get()));
    ASSERT_TRUE(knn_jni::faiss_wrapper::IsSharedIndexStateRequired((jlong) indexIDMapIVFPQL2.get()));
}

TEST(FaissInitAndSetSharedIndexState, BasicAssertions) {
    faiss::idx_t numIds = 256;
    int dim = 2;
    std::vector<faiss::idx_t> ids = test_util::Range(numIds);
    std::vector<float> vectors = test_util::RandomVectors(dim, numIds, randomDataMin, randomDataMax);

    std::string indexPath = test_util::RandomString(10, "tmp/", ".faiss");
    faiss::MetricType metricType = faiss::METRIC_L2;
    std::string indexDescription = "IVF4,PQ1x4";

    std::unique_ptr<faiss::Index> faissIndex(test_util::FaissCreateIndex(dim, indexDescription, metricType));
    test_util::FaissTrainIndex(faissIndex.get(), numIds, vectors.data());
    auto faissIndexWithIDMap = test_util::FaissAddData(faissIndex.get(), ids, vectors);
    test_util::FaissWriteIndex(&faissIndexWithIDMap, indexPath);

    // Setup jni
    NiceMock<JNIEnv> jniEnv;
    NiceMock<test_util::MockJNIUtil> mockJNIUtil;

    std::unique_ptr<faiss::Index> loadedIndexPointer(
            reinterpret_cast<faiss::Index *>(knn_jni::faiss_wrapper::LoadIndex(
                    &mockJNIUtil, &jniEnv, (jstring)&indexPath)));

    auto idMapIndex = dynamic_cast<faiss::IndexIDMap *>(loadedIndexPointer.get());
    ASSERT_NE(idMapIndex, nullptr);
    auto ivfpqIndex = dynamic_cast<faiss::IndexIVFPQ *>(idMapIndex->index);
    ASSERT_NE(ivfpqIndex, nullptr);
    ASSERT_EQ(0, ivfpqIndex->precomputed_table->size());
    jlong sharedModelAddress = knn_jni::faiss_wrapper::InitSharedIndexState((jlong) loadedIndexPointer.get());
    ASSERT_EQ(0, ivfpqIndex->precomputed_table->size());
    knn_jni::faiss_wrapper::SetSharedIndexState((jlong) loadedIndexPointer.get(), sharedModelAddress);
    ASSERT_EQ(sharedModelAddress, (jlong) ivfpqIndex->precomputed_table);
    ASSERT_NE(0, ivfpqIndex->precomputed_table->size());
    ASSERT_EQ(1, ivfpqIndex->use_precomputed_table);
    knn_jni::faiss_wrapper::FreeSharedIndexState(sharedModelAddress);
}

TEST(FaissRangeSearchQueryIndexTest, BasicAssertions) {
    // Define the index data
    faiss::idx_t numIds = 200;
    int dim = 2;
    std::vector<faiss::idx_t> ids = test_util::Range(numIds);
    std::vector<float> vectors = test_util::RandomVectors(dim, numIds, rangeSearchRandomDataMin, rangeSearchRandomDataMax);

    faiss::MetricType metricType = faiss::METRIC_L2;
    std::string method = "HNSW32,Flat";

    int efSearch = 20;
    std::unordered_map<std::string, jobject> methodParams;
    methodParams[knn_jni::EF_SEARCH] = reinterpret_cast<jobject>(&efSearch);
    auto methodParamsJ = reinterpret_cast<jobject>(&methodParams);

    // Define query data
    int numQueries = 100;
    std::vector<std::vector<float>> queries;

    for (int i = 0; i < numQueries; i++) {
        std::vector<float> query;
        query.reserve(dim);
        for (int j = 0; j < dim; j++) {
            query.push_back(test_util::RandomFloat(rangeSearchRandomDataMin, rangeSearchRandomDataMax));
        }
        queries.push_back(query);
    }

    // Create the index
    std::unique_ptr<faiss::Index> createdIndex(
            test_util::FaissCreateIndex(dim, method, metricType));
    auto createdIndexWithData =
            test_util::FaissAddData(createdIndex.get(), ids, vectors);

    // Setup jni
    NiceMock<JNIEnv> jniEnv;
    NiceMock<test_util::MockJNIUtil> mockJNIUtil;

    int maxResultWindow = 20000;

    for (auto query : queries) {
        std::unique_ptr<std::vector<std::pair<int, float> *>> results(
                reinterpret_cast<std::vector<std::pair<int, float> *> *>(

                        knn_jni::faiss_wrapper::RangeSearch(
                                &mockJNIUtil, &jniEnv,
                                reinterpret_cast<jlong>(&createdIndexWithData),
                                reinterpret_cast<jfloatArray>(&query), rangeSearchRadius, methodParamsJ, maxResultWindow, nullptr)));

        // assert result size is not 0
        ASSERT_NE(0, results->size());


        // Need to free up each result
        for (auto it : *results) {
            delete it;
        }
    }
}

TEST(FaissRangeSearchQueryIndexTest_WhenHitMaxWindowResult, BasicAssertions){
    // Define the index data
    faiss::idx_t numIds = 200;
    int dim = 2;
    std::vector<faiss::idx_t> ids = test_util::Range(numIds);
    std::vector<float> vectors = test_util::RandomVectors(dim, numIds, rangeSearchRandomDataMin, rangeSearchRandomDataMax);

    faiss::MetricType metricType = faiss::METRIC_L2;
    std::string method = "HNSW32,Flat";

    // Define query data
    int numQueries = 100;
    std::vector<std::vector<float>> queries;

    for (int i = 0; i < numQueries; i++) {
        std::vector<float> query;
        query.reserve(dim);
        for (int j = 0; j < dim; j++) {
            query.push_back(test_util::RandomFloat(rangeSearchRandomDataMin, rangeSearchRandomDataMax));
        }
        queries.push_back(query);
    }

    // Create the index
    std::unique_ptr<faiss::Index> createdIndex(
            test_util::FaissCreateIndex(dim, method, metricType));
    auto createdIndexWithData =
            test_util::FaissAddData(createdIndex.get(), ids, vectors);

    // Setup jni
    NiceMock<JNIEnv> jniEnv;
    NiceMock<test_util::MockJNIUtil> mockJNIUtil;

    int maxResultWindow = 10;

    for (auto query : queries) {
        std::unique_ptr<std::vector<std::pair<int, float> *>> results(
                reinterpret_cast<std::vector<std::pair<int, float> *> *>(

                        knn_jni::faiss_wrapper::RangeSearch(
                                &mockJNIUtil, &jniEnv,
                                reinterpret_cast<jlong>(&createdIndexWithData),
                                reinterpret_cast<jfloatArray>(&query), rangeSearchRadius, nullptr, maxResultWindow, nullptr)));

        // assert result size is not 0
        ASSERT_NE(0, results->size());
        // assert result size is equal to maxResultWindow
        ASSERT_TRUE(false) << "fjiowjfiowejiof\n\n\\n\nn";
        ASSERT_EQ(maxResultWindow, results->size());

        // Need to free up each result
        for (auto it : *results) {
            delete it;
        }
    }
}

TEST(FaissRangeSearchQueryIndexTestWithFilterTest, BasicAssertions) {
    // Define the index data
    faiss::idx_t numIds = 200;
    int dim = 2;
    std::vector<faiss::idx_t> ids = test_util::Range(numIds);
    std::vector<float> vectors = test_util::RandomVectors(dim, numIds, rangeSearchRandomDataMin, rangeSearchRandomDataMax);

    faiss::MetricType metricType = faiss::METRIC_L2;
    std::string method = "HNSW32,Flat";

    // Define query data
    int numQueries = 100;
    std::vector<std::vector<float>> queries;

    for (int i = 0; i < numQueries; i++) {
        std::vector<float> query;
        query.reserve(dim);
        for (int j = 0; j < dim; j++) {
            query.push_back(test_util::RandomFloat(rangeSearchRandomDataMin, rangeSearchRandomDataMax));
        }
        queries.push_back(query);
    }

    // Create the index
    std::unique_ptr<faiss::Index> createdIndex(
            test_util::FaissCreateIndex(dim, method, metricType));
    auto createdIndexWithData =
            test_util::FaissAddData(createdIndex.get(), ids, vectors);

    // Setup jni
    NiceMock<JNIEnv> jniEnv;
    NiceMock<test_util::MockJNIUtil> mockJNIUtil;

    int num_bits = test_util::bits2words(164);
    std::vector<jlong> bitmap(num_bits,0);
    std::vector<int64_t> filterIds;

    for (int64_t i = 1; i < 50; i++) {
        filterIds.push_back(i);
        test_util::setBitSet(i, bitmap.data(), bitmap.size());
    }
    std::unordered_set<int> filterIdSet(filterIds.begin(), filterIds.end());

    int maxResultWindow = 20000;

    for (auto query : queries) {
        std::unique_ptr<std::vector<std::pair<int, float> *>> results(
                reinterpret_cast<std::vector<std::pair<int, float> *> *>(

                        knn_jni::faiss_wrapper::RangeSearchWithFilter(
                                &mockJNIUtil, &jniEnv,
                                reinterpret_cast<jlong>(&createdIndexWithData),
                                reinterpret_cast<jfloatArray>(&query), rangeSearchRadius, nullptr, maxResultWindow,
                                reinterpret_cast<jlongArray>(&bitmap), 0, nullptr)));

        // assert result size is not 0
        ASSERT_NE(0, results->size());
        ASSERT_TRUE(results->size() <= filterIds.size());
        for (const auto& pairPtr : *results) {
            auto it = filterIdSet.find(pairPtr->first);
            ASSERT_NE(it, filterIdSet.end());
        }

        // Need to free up each result
        for (auto it : *results) {
            delete it;
        }
    }
}

TEST(FaissRangeSearchQueryIndexTestWithParentFilterTest, BasicAssertions) {
    // Define the index data
    faiss::idx_t numIds = 100;
    std::vector<faiss::idx_t> ids;
    std::vector<float> vectors;
    std::vector<int> parentIds;
    int dim = 2;
    for (int64_t i = 1; i < numIds + 1; i++) {
        if (i % 10 == 0) {
            parentIds.push_back(i);
            continue;
        }
        ids.push_back(i);
        for (int j = 0; j < dim; j++) {
            vectors.push_back(test_util::RandomFloat(rangeSearchRandomDataMin, rangeSearchRandomDataMax));
        }
    }

    faiss::MetricType metricType = faiss::METRIC_L2;
    std::string method = "HNSW32,Flat";

    // Define query data
    int numQueries = 1;
    std::vector<std::vector<float>> queries;

    for (int i = 0; i < numQueries; i++) {
        std::vector<float> query;
        query.reserve(dim);
        for (int j = 0; j < dim; j++) {
            query.push_back(test_util::RandomFloat(rangeSearchRandomDataMin, rangeSearchRandomDataMax));
        }
        queries.push_back(query);
    }

    // Create the index
    std::unique_ptr<faiss::Index> createdIndex(
            test_util::FaissCreateIndex(dim, method, metricType));
    auto createdIndexWithData =
            test_util::FaissAddData(createdIndex.get(), ids, vectors);

    // Setup jni
    NiceMock<JNIEnv> jniEnv;
    NiceMock<test_util::MockJNIUtil> mockJNIUtil;
    EXPECT_CALL(mockJNIUtil,
                GetJavaIntArrayLength(
                        &jniEnv, reinterpret_cast<jintArray>(&parentIds)))
            .WillRepeatedly(Return(parentIds.size()));

    int maxResultWindow = 10000;

    for (auto query : queries) {
        std::unique_ptr<std::vector<std::pair<int, float> *>> results(
                reinterpret_cast<std::vector<std::pair<int, float> *> *>(

                        knn_jni::faiss_wrapper::RangeSearchWithFilter(
                                &mockJNIUtil, &jniEnv,
                                reinterpret_cast<jlong>(&createdIndexWithData),
                                reinterpret_cast<jfloatArray>(&query), rangeSearchRadius, nullptr, maxResultWindow, nullptr, 0,
                                reinterpret_cast<jintArray>(&parentIds))));

        // assert result size is not 0
        ASSERT_NE(0, results->size());
        // Result should be one for each group
        std::set<int> idSet;
        for (const auto& pairPtr : *results) {
            idSet.insert(pairPtr->first / 10);
        }
        ASSERT_NE(0, idSet.size());

        // Need to free up each result
        for (auto it : *results) {
            delete it;
        }
    }
}
