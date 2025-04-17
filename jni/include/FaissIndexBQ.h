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

#ifndef KNNPLUGIN_JNI_FAISSINDEXBQ_H
#define KNNPLUGIN_JNI_FAISSINDEXBQ_H

#include "faiss/IndexFlatCodes.h"
#include "faiss/Index.h"
#include "faiss/impl/DistanceComputer.h"
#include "faiss/utils/hamming_distance/hamdis-inl.h"
#include <vector>
#include <iostream>
#include <format>

namespace knn_jni {
    namespace faiss_wrapper {
        struct CustomerFlatCodesDistanceComputer : faiss::FlatCodesDistanceComputer {
            const float* query;
            int dimension;
            size_t code_size;
            faiss::MetricType metric_type;

            CustomerFlatCodesDistanceComputer(const uint8_t* codes, size_t code_size, int d, 
                faiss::MetricType metric_type = faiss::METRIC_L2) 
            // : FlatCodesDistanceComputer(codes, code_size), dimension(d), query(nullptr) {
                : FlatCodesDistanceComputer(codes, code_size), dimension(d), query(nullptr), metric_type(metric_type) {
                this->codes = codes;
                this->code_size = code_size;
                this->dimension = d;
            }

            float distance_to_code_l2(const uint8_t* code) {
                 // L2 distance
                float score = 0.0f;
                for (int i = 0; i < dimension; i++) {
                    uint8_t code_block = code[(i / 8)];
                    int bit_offset = 7 - (i % 8);
                    // int bit_offset = i % 8;
                    int bit_mask = 1 << bit_offset;
                    int code_masked = (code_block & bit_mask);
                    int code_translated = code_masked >> bit_offset;

                    // want to select the
                    // std::cout << "bit_offset: " << bit_offset << std::endl;
                    // std::cout << "bit_mask: " << bit_mask << std::endl;
                    // std::cout << "code_masked: " << code_masked << std::endl;
                    // std::cout << "code_translated: " << code_translated << std::endl;

                    // Inner product
                    // float dim_score = code_translated == 0 ? 0 : -1*query[i];

                    // L2
                    float dim_score = (code_translated - query[i]) * (code_translated - query[i]);

                    score += dim_score;
                }
                return score;
                    
            }
            float distance_to_code_IP(const uint8_t* code) {
                // Inner product distance
                float score = 0.0f;
                for (int i = 0; i < dimension; i++) {
                    uint8_t code_block = code[(i / 8)];
                    int bit_offset = 7 - (i % 8);
                    int bit_mask = 1 << bit_offset;
                    int code_masked = (code_block & bit_mask);
                    int code_translated = code_masked >> bit_offset;

                    // want to select the
                    // std::cout << "bit_offset: " << bit_offset << std::endl;
                    // std::cout << "bit_mask: " << bit_mask << std::endl;
                    // std::cout << "code_masked: " << code_masked << std::endl;
                    // std::cout << "code_translated: " << code_translated << std::endl;

                    // Inner product
                    // float dim_score = code_translated == 0 ? 0 : -1*query[i];

                    // inner product
                    float dim_score = code_translated == 0 ? 0 : -1*query[i];
                    // float dim_score = code_translated * query[i];

                    score += dim_score;
                }
                return -score; // TODO weird things about negation in faiss and with innerproducts. 
            };

            virtual float distance_to_code(const uint8_t* code) override {

                // std::cout << "distance to code called, normalied query vector: ";

                // for (int i = 0; i < this->dimension; i++) { 
                //     std::cout << query[i] << " ";
                // }

                // std::cout << "\n END QUERY VECTOR \n";

                // Compute the dot product between the 2
                // TODO: How can we do this better for 2-bit and 4-bit
                // I think we would want to just shift the multiplier of 1. i.e.
                // -1 << 1 *query[i]
                // -1 << 2 *query[i]
                // -1 << 3 *query[i]
                // Debug print first few values
                // for (int i = 0; i < std::min(dimension, 32); i++) {
                    // std::cout << "bit " << i << ": " 
                            // << ((codes[i / 8] & (1 << (i % 8))) != 0) << std::endl;
                // }
                // std::cout << "called here\n\n\n";
//                 float score = 0.0f;

//                 // std::cout << " code: " << static_cast<int>(code[0]);
//                 for (int i = 0; i < this->dimension; i++) {
//                     // score += (code[(i / sizeof(uint8_t))] & (1 << (i % sizeof(uint8_t)))) == 0 ? -1 * query[i] * query[i] : -1 * (1- query[i]) * (1-query[i]);
//                     // score += (code[(i / sizeof(uint8_t))] & (1 << (i % sizeof(uint8_t)))) == 0 ? 0 : -1*query[i];
//                     // score += (code[(i / sizeof(uint8_t))] & (1 << (i % sizeof(uint8_t)))) == 0 ? query[i] * query[i] : (1- query[i]) * (1-query[i]);
//                     // float code_val = (code[(i / 8)] & (1 << (i % 8))) ? 1.0f : 0.0f;
//                     // float code_val = (code[(i / 8)] & (1 << (7 - (i % 8)))) ? 1.0f : 0.0f;
//                 //    std::cout << ((code[(i / 8)] & (1 << (i % 8))) != 0);
//                 //    if (i % 32 == 31) std::cout << "\n";
//                     float code_val = (code[(i / 8)] & ((1 << (i % 8)))) ? 1.0f : 0.0f;
//                     // score += 1 - 2 * (query[i] - code_val);
//                     // score += (code[(i / 8)] & ((1 << (i % 8)))) == 0 ? (query[i] ) * (query[i]) : (query[i] - 1) * (query[i] - 1);
//                     score += ((code[(i / 8)] & ((1 << (i % 8)))) >> (i % 8)) == 0 ? 0 : query[i];
//                     // score += (code[(i / (8 * sizeof(uint8_t)))] & (1 << (i % (8 * sizeof(uint8_t))))) == 0 ? query[i] * query[i] : (1- query[i]) * (1-query[i]);
//                 }
// //                return std::sqrt(score);
// //                std::cout << "score: " << score;
//                 return score;
    

    // TODO make this less hacky; we're going to change things to use innerproduct next. 

    // Might want a better test suite than the preexisting so that I can verify that innerproduct distances are correct.
    // probably just a float vector with 1s and 0s. Confirm that it's the same 
                if (this->metric_type == faiss::METRIC_L2) {
                    return distance_to_code_l2(code);
                } else if (this->metric_type == faiss::METRIC_INNER_PRODUCT) {
                    return distance_to_code_IP(code);
                } else {
std::cout << ("ADC distance computer called with unsupported metric, see faiss;:MetricType enum w metric" + std::to_string(this->metric_type)); // would be nice to have std::format w C++20....
                    throw std::runtime_error(
                        ("ADC distance computer called with unsupported metric, see faiss;:MetricType enum w metric" + std::to_string(this->metric_type)) // would be nice to have std::format w C++20....
                    );
                }
   
    // // Inner product distance
    // float score = 0.0f;
    // for (int i = 0; i < dimension; i++) {
    //     uint8_t code_block = code[(i / 8)];
    //     int bit_offset = 7 - (i % 8);
    //     int bit_mask = 1 << bit_offset;
    //     int code_masked = (code_block & bit_mask);
    //     int code_translated = code_masked >> bit_offset;

    //     // want to select the
    //     // std::cout << "bit_offset: " << bit_offset << std::endl;
    //     // std::cout << "bit_mask: " << bit_mask << std::endl;
    //     // std::cout << "code_masked: " << code_masked << std::endl;
    //     // std::cout << "code_translated: " << code_translated << std::endl;

    //     // Inner product
    //     // float dim_score = code_translated == 0 ? 0 : -1*query[i];

    //     // inner product
    //     float dim_score = code_translated == 0 ? 0 : -1*query[i];

    //     score += dim_score;
    // }
    // return score; // TODO weird things about negation in faiss and with innerproducts. 
};



            virtual void set_query(const float* x) override {
                this->query = x;
            };

            virtual float symmetric_dis(faiss::idx_t i, faiss::idx_t j) override {
                std::cout << " in hamming sym dist for some reason...";

                // Just return hamming distance for now...
                return faiss::hamming<1, float>(&this->codes[i], &this->codes[j]);
            };
        };

        struct FaissIndexBQ : faiss::IndexFlatCodes {

            FaissIndexBQ(faiss::idx_t d, std::vector<uint8_t> codes, faiss::MetricType metric=faiss::METRIC_L2) 
            : IndexFlatCodes(d/8, d, metric){
                // std::cout << "FaissIndexBQ constructor called with codes lenght" << codes.size() << "and codes 0\n" << " and d/8 " << d/8 << " and d " << d << " and metric" << metric;
//                << codes[0] << "\n";
                // std::cout << "\nHEREHERHERH\n\n\n\n\n\n\n\n\n";
                // this->d = d;
                this->codes = codes;
                this->code_size = (d/8);
                // this->code_size = 16;
            }

            void init(faiss::Index * parent, faiss::Index * grand_parent) {
                // std::cout << "ehreheragainga\n\n\n\n";
                this->ntotal = this->codes.size() / (this->d / 8);
                parent->ntotal = this->ntotal;   
                grand_parent->ntotal = this->ntotal;
            }

            /** a FlatCodesDistanceComputer offers a distance_to_code method */
            faiss::FlatCodesDistanceComputer* get_FlatCodesDistanceComputer() const override {
                // std::cout << "number of codes: " << this->codes.size() << "\n\n\n HEREHERHEHEREHRHEHRUIHWEUIFHIU\n\\n\n\n\n\n"; // 4400
            //    std::cout << "0th code: " << static_cast<int>(this->codes[0]) << "\n";
                // std::cout << "ntotal: " << this->ntotal << "\n";
                // std::cout << "code sz: " << this->code_size << "\n" << std::endl;
                // std::cout << "LOOK HERE!!!\n\n" << this->metric_type << "\n\nLOOK HERE!!!\n\n" << std::endl;
            //    std::cout << this->d << "\n";

            //    for (uint8_t code : this->codes) {
            //        std::cout << static_cast<int>(code) << " ";
            //    }
// faiss::METRIC_INNER_PRODUCT
                return new knn_jni::faiss_wrapper::CustomerFlatCodesDistanceComputer((const uint8_t*) (this->codes.data()), this->d/8, this->d,
            this->metric_type);
                // TODO make the code_size calculation better, including rounding up via (d + 7) / 8
            };

            // virtual void merge_from(faiss::Index& otherIndex, faiss::idx_t add_id = 0) override {
            //     IndexFlatCodes::merge_from(otherIndex, add_id);
            // };

            // virtual void search(
            //         faiss::idx_t n,
            //         const float* x,
            //         faiss::idx_t k,
            //         float* distances,
            //         faiss::idx_t* labels,
            //         const faiss::SearchParameters* params = nullptr) const override {};
                    // {
                    //     IndexFlatCodes::search(n,x,k,distances,labels,params);
                    // };
        };
    }
}
#endif //KNNPLUGIN_JNI_FAISSINDEXBQ_H
