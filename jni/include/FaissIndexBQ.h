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
#include <cassert>

namespace knn_jni {
    namespace faiss_wrapper {
        struct CustomerFlatCodesDistanceComputer : faiss::FlatCodesDistanceComputer {
            const float* query;
            int dimension;
            size_t code_size;
            faiss::MetricType metric_type;
            std::vector<std::vector<float>> lookup_table;
            std::vector<float> coord_scores; // scores for each dimension
            float correction_amount; // used in batched distances

            CustomerFlatCodesDistanceComputer(const uint8_t* codes, size_t code_size, int d, 
                faiss::MetricType metric_type = faiss::METRIC_L2) 
                : FlatCodesDistanceComputer(codes, code_size), dimension(d), query(nullptr), metric_type(metric_type) {
                this->codes = codes;
                this->code_size = code_size;
                this->dimension = d;
                correction_amount = 0.0f;
            }

            float distance_to_code_batched(const uint8_t * code) {
                float dist = 0.0f; // dist = this->query_correction;
                for (int i = 0 ; i < dimension / 8; ++i) {
                    const unsigned char code_batch = code[i];
                    // std::cout << "access lookup table";
                    dist += this->lookup_table[i][code_batch];
                }

                return dist + correction_amount; 
            }

            virtual float distance_to_code(const uint8_t* code) override {
    // TODO make this less hacky; we're going to change things to use innerproduct next. 

    // Might want a better test suite than the preexisting so that I can verify that innerproduct distances are correct.
    // probably just a float vector with 1s and 0s. Confirm that it's the same 
                return distance_to_code_batched(code);
    
//     if (this->metric_type == faiss::METRIC_L2) {
//                     return distance_to_code_l2_batched(code);
//                 } else if (this->metric_type == faiss::METRIC_INNER_PRODUCT) {
//                     return distance_to_code_IP(code);
//                 } else {
// std::cout << ("ADC distance computer called with unsupported metric, see faiss;:MetricType enum w metric" + std::to_string(this->metric_type)); // would be nice to have std::format w C++20....
//                     throw std::runtime_error(
//                         ("ADC distance computer called with unsupported metric, see faiss;:MetricType enum w metric" + std::to_string(this->metric_type)) // would be nice to have std::format w C++20....
//                     );
//                 }
};

            void compute_cord_scores() {
                assert(this->query != nullptr); // make sure we've already set the query
                this->coord_scores = std::vector<float>(this->dimension, 0.0f);
                if (this->metric_type == faiss::METRIC_L2) {
                    compute_cord_scores_l2(); // todo make this templated based on space type.
                } else if (this->metric_type == faiss::METRIC_INNER_PRODUCT) {
                    compute_cord_scores_inner_product(); 
                }
                
                else {
        std::cout << ("ADC distance computer called with unsupported metric, see faiss;:MetricType enum w metric" + std::to_string(this->metric_type)); // would be nice to have std::format w C++20....
                            throw std::runtime_error(
                                ("ADC distance computer called with unsupported metric, see faiss;:MetricType enum w metric" + std::to_string(this->metric_type)) // would be nice to have std::format w C++20....
                            );
                        }
            }

            void compute_cord_scores_l2() {
                assert(query != nullptr);
                for (int i = 0 ; i < this->dimension; ++i) {
                    float x = query[i];
                    this->coord_scores[i] = 1 - 2 * x;
                    correction_amount += x * x;
                }
            }

            void compute_cord_scores_inner_product() {
                // assert(query != nullptr);
                for (int i = 0 ; i < this->dimension; ++i) {
                    // float x =;
                    this->coord_scores[i] = query[i];
                    // correction_amount += x * x;
                }
            }



            virtual void set_query(const float* x) override {
                this->query = x;
                compute_cord_scores();
                create_batched_lookup_table();
            };
            // compute all possible distance combinations for the batch at batch_idx against our query vector
            void compute_per_batch_lookup(int batch_idx, std::vector<float> & batch) {
                // assert(batch.size() == 256); // TODO magic constants 
                // assert(this->query != nullptr); // make sure we've already set the query
                // batch has 256 dimension, and it only looks at one 8-bit/1 byte chunk of the query vector.
                for (int i = 0 ; i < 8; ++i) {
                    const unsigned int bit_masked = 1 << i;
                    const float bit_value = this->coord_scores[ batch_idx * 8 // for instance for batch_idx 1, this looks starting at position 7
                      + (7 - i) // and then scans from right to left. TODO might not want this reversal...
                    ]; // lookup the particular value if this bit is 1 from within the coord scores. 

                    for (unsigned int suffix = 0; suffix < bit_masked; ++suffix) {
                        batch[
                            bit_masked | suffix // range from 0 to 255
                        ] = batch[suffix] + bit_value;
                    }

                }
            }

            void create_batched_lookup_table() {
                const unsigned int num_batches =this->dimension/8; // how many batches per vector
                this->lookup_table = std::vector<std::vector<float>>( num_batches, std::vector<float>( 256, 0.0f));

                    // [this->dimension/8][256]; // TODO magic constants
                    
                for (int i = 0 ; i < num_batches; ++i) {
                    compute_per_batch_lookup(i, this->lookup_table[i]);
                }
            }

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
            //    std::cout << this->d << "\n";f 

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
