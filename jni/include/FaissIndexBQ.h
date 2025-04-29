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

        // here add 2 bit distance computer

        struct ADCFlatCodesDistanceComputer2Bit : faiss::FlatCodesDistanceComputer {
            const float* query;
            int dimension;
            size_t code_size;
            faiss::MetricType metric_type;
            std::vector<std::vector<float>> lookup_table; // used in batched distances
            std::vector<float> coord_scores; // scores for each dimension
            float correction_amount; // used in batched distances

            std::vector<std::vector<float>> partitions;                
            std::vector<float> above_threshold_means;
            std::vector<float> below_threshold_means;

            std::vector<float> z;
            // TODO note: the dimension passed in from k-NN is the number of bits
            // so for 2 bit quant, and 128 dimension vectors, d = 256. 
            ADCFlatCodesDistanceComputer2Bit(const uint8_t * codes, size_t code_size, int d, faiss::MetricType metric_type = faiss::METRIC_L2,
                std::vector<float> above_threshold_means = std::vector<float>()
                , std::vector<float> below_threshold_means = std::vector<float>()
            )
            : FlatCodesDistanceComputer(codes, code_size), dimension(d), query(nullptr), metric_type(metric_type) {
                // std::cout << "faiss uq 2 bit distance computer ctor  " << std::endl;
                this->codes = codes;
                this->code_size = code_size;
                this->dimension = d/2; // since the underlying index has dimension as 2 * vector_dim. 
                this->correction_amount = 0.0f; 
                this->above_threshold_means = above_threshold_means;
                
                this->below_threshold_means = below_threshold_means;
                // std::cout << " above thres first elt " << this->above_threshold_means[0] << std::endl;
                // std::cout << "this->correction_amoutn: " << std::to_string(this->correction_amount) << std::endl;

                // std::cout << "code size : " << this->code_size << " dimensions: " << this->dimension;
                this->partitions = std::vector<std::vector<float>>(
                    this->dimension, std::vector<float> (3 , 0.0f) // TODO magic constant
                );


                for (int i = 0; i < this->dimension; ++i) {
                    float x = below_threshold_means[i];
                    float y = above_threshold_means[i]; 
                    partitions[i][0] = x;
                    partitions[i][1] = (x + y) / 2.0f;
                    partitions[i][2] = y;
                }

                this->z = std::vector<float>(this->dimension*2, 0.0f ); // we want it to be d * 2

            }

            virtual float distance_to_code(const uint8_t* code) override {
                // std::cout << " dist to code " << this->above_threshold_means[0] << std::endl;
                // code has dimension * 2 bits. So (dimension * 2 / 8) bytes
                // q: 2.0 -3.0 4.0 -9.0 
                // p: 00 10
                // d = 2
                // q * p = 4.0 
                //
                // return 0.0f;
                // std::cout << "z (sz) " << z.size() << "(distance to code): ";
                // for (int j = 0; j < (this->z).size(); ++j) {
                //     std::cout << this->z[j] << " ";
                // }
                // std::cout << std::endl;
                
                // tbe java bit packing code sets the first bit for all dimensions, then the second bit for all dimensions, etc.
                // 
                // compute p dot z
                float distance = 0.0f; 
                for (int code_byte_idx = 0; code_byte_idx < this->dimension / 8; ++code_byte_idx) {
                    // std::cout << "before code access " << std::endl;
                    uint8_t first_code_byte = code[code_byte_idx];
                    
                    // the matching bits for this document are at code_byte[code_bit], (code_byte+this->dimension/8)[code_bit] I think.
// std::cout << "before second code access " << std::endl;
                    uint8_t second_code_byte = code[(this->dimension / 8) + code_byte_idx];
                    for (int first_code_bit = 0; first_code_bit < 8; ++first_code_bit) {
                        
                        int first_code_entry = (first_code_byte >> (7 - first_code_bit)) & 1;
                        int second_code_entry = (second_code_byte >> (7 - first_code_bit)) & 1;

                        int z_idx = code_byte_idx * 16 + first_code_bit * 2;
                        // std::cout << "before first z access " << std::endl;
                        float first_z_val = z[
                            z_idx
                        ];
                        float second_z_val = z[
                            z_idx + 1
                        ];

                        // int code_entry = first_code_entry + second_code_entry;

                        if (first_code_entry == 1) {
                            
                            if (second_code_entry == 0) { // 10
                                distance += first_z_val;
                            } else { // 11
                                distance += first_z_val + second_z_val;
                            }
                        }
                        // if (code_entry == 0) { // 00
                            // distance += 0; // noop
                        // }
                        // else if (code_entry == 1) {// 01
                            
                        //     if (first_code_entry == 1) { // 10
                        //         // distance += second_z_val;
                        //         distance += first_z_val;
                        //         // std::cout << "invalid code for 2 bit unary qunatization (can't have 1 before 0), code: " 
                        //         //     + std::to_string(first_code_entry) + std::to_string(second_code_entry) << " at idx " << code_byte_idx << " "
                        //         //     << " first code bit " << first_code_bit << std::endl;
                        //         // throw std::runtime_error("invalid code for 2 bit unary qunatization (can't have 1 before 0), code: " 
                        //         //     + std::to_string(first_code_entry) + std::to_string(second_code_entry));
                        //     } else {
                        //         // 01 
                        //         // distance += first_z_val;
                        //         distance += second_z_val;
                        //         std::cout << "okay code for 2 bit unary qunatization code: " 
                        //             + std::to_string(first_code_entry) + std::to_string(second_code_entry) << " at idx " << code_byte_idx << " "
                        //             << " first code bit " << first_code_bit << std::endl;
                        //     }
                            
                        // } 
                        // else if (code_entry == 2) { // 11
                        //     distance += (first_z_val + second_z_val);
                        // }
                        // else {
                        //     throw std::runtime_error("invalid code for 2 bit unary qunatization, code: " 
                        //         + std::to_string(first_code_entry) + std::to_string(second_code_entry));
                        // }
                    }
                }
                // std::cout << "just before return " << std::endl;
                // std::cout << "distance: " << std::to_string(distance) << std::endl;
                // std::cout << "correction amount: " << std::to_string(this->correction_amount) << std::endl;
                // std::cout << "sum: " << std::to_string(distance + this->correction_amount) << std::endl;
                return distance + this->correction_amount;




                //     for (int bit_pair = 0; bit_pair < 4; ++bit_pair) {
                        
                //         int shift_amount = 6 - bit_pair * 2; // 6, 4, 2, 0
                //         int code_entry = (code_byte >> shift_amount) & 0b11;


                //         int byte_one;

                //         int byte_two; // it's over by dimension/8 for 2 bit case

                //         int dim_idx = code_byte_idx * 4 + bit_pair;
                //         float a = z[dim_idx * 2];
                //         float b = z[dim_idx * 2 + 1];
                //         switch (code_entry) { //
                //             case 0: // 00
                //                 distance += 0;
                //                 break;
                //             case 1: // 01
                //                 distance += a;
                //                 // std::cout << "01 found" << std::endl;
                //                 break;
                //             case 3: // 11
                //                 distance += (a + b);
                //                 break;
                //             case 2: // 10, not valid?
                //                 throw std::runtime_error("invalid code for 2 bit unary qunatization, code: " + code_entry);
                //                 distance += b;
                //                 break;
                //             default:

                //                 // std::cout << " weird code called for code byte index " << code_byte_idx << " \n";
                //                 std::string error_msg = "invalid code for 2 bit unary qunatization, code: " + std::to_string(code_entry);
                //                 throw std::runtime_error(error_msg);
                //                 break;
                            
                //         }

                //         // code_byte = code_byte >> 2; // investigate next 2 bits
                //     }
                // }

                // return dot_product + this->correction_amount;

                // return distance + this->correction_amount;
            };

            void compute_z() {
                // std::cout << "this dimension: " << this->dimension << std::endl;
                std::fill(z.begin(), z.end(), 0.0f);
                // this->z = std::vector(this->dimension * 2, 0.0f ); // reset z to be we want it to be d * 2
                for (int i = 0 ; i < this->dimension; ++i) {
                    // correction_amount +=  // first bit (additive error)
                    // TODO make this more efficient with a good array access pattern after I get poc working.
                    
                    float c = this->query[i];
                    float first_partition_distance = (c - this->partitions[i][0]) * (c - this->partitions[i][0]);

                    float accumulator = first_partition_distance;

                    this->correction_amount += first_partition_distance; 

                    for (int partition_idx = 1; partition_idx < 3; partition_idx ++) { // TODO 
                        float m = this->partitions[i][partition_idx];

                        float v = (c - m) * (c - m);

                        float d = v - accumulator;
                        z[i * 2 + partition_idx-1] = d;
                        
                        accumulator += d;
                    }
                }


                
                // std::cout << "z: ";
                // for (int j = 0; j < z.size(); ++j) {
                //     std::cout << z[j] << " ";
                // }
                // std::cout << std::endl;
            }

            virtual void set_query(const float* x) override {
                // std::cout << " query set " << this->below_threshold_means[0] << std::endl;
                this->correction_amount = 0.0f;
                this->query = x;
                // vcompute z                
                compute_z();
            };

            virtual float symmetric_dis(faiss::idx_t i, faiss::idx_t j) override {
                throw std::runtime_error("symmetric distance should not be called for the adc 2 bit distance computer during indexing time");
            }
        };



        struct ADCFlatCodesDistanceComputer4Bit : faiss::FlatCodesDistanceComputer {
            const float* query;
            int dimension;
            size_t code_size;
            faiss::MetricType metric_type;
            std::vector<std::vector<float>> lookup_table; // used in batched distances
            std::vector<float> coord_scores; // scores for each dimension
            float correction_amount; // used in batched distances

            std::vector<std::vector<float>> partitions;                
            std::vector<float> above_threshold_means;
            std::vector<float> below_threshold_means;

            ADCFlatCodesDistanceComputer4Bit(const uint8_t * codes, size_t code_size, int d, faiss::MetricType metric_type = faiss::METRIC_L2,
                std::vector<float> above_threshold_means= std::vector<float>(), std::vector<float> below_threshold_means = std::vector<float>()
            )
            : FlatCodesDistanceComputer(codes, code_size), dimension(d), query(nullptr), metric_type(metric_type) {
                this->codes = codes;
                this->code_size = code_size;
                this->dimension = d;
                correction_amount = 0.0f; 

            }

            virtual float distance_to_code(const uint8_t* code) override {
                return 0.0f;
            };

            virtual void set_query(const float* x) override {
                correction_amount = 0.0f;
                this->query = x;
                // vcompute z
                // here we need to calculate the partitions. 
            };

            virtual float symmetric_dis(faiss::idx_t i, faiss::idx_t j) override {
                throw std::runtime_error("symmetric distance should not be called for the adc 2 bit distance computer during indexing time");
            }
        };


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
                correction_amount = 0.0f;
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
    

    struct FaissIndexUQ2Bit : faiss::IndexFlatCodes {
        // as part of the input, pass in a partitions vector .
        std::vector<float> above_threshold_means;
        std::vector<float> below_threshold_means;

        FaissIndexUQ2Bit(
            faiss::idx_t d, std::vector<uint8_t> codes, faiss::MetricType metric=faiss::METRIC_L2, std::vector<float> above_threshold_mean_vector = std::vector<float>(), std::vector<float> below_threshold_mean_vector= std::vector<float>()
        ) : IndexFlatCodes(d/8, d, metric){
            // std::cout << "faiss uq 2 bit ctor , dimension " << d << "."  << std::endl;
            this->codes = codes; 
            this->code_size = (d/ 8);
            this->above_threshold_means = above_threshold_mean_vector;
            this->below_threshold_means = below_threshold_mean_vector;
        }

        void init(faiss::Index * parent, faiss::Index * grand_parent) {
            // std::cout << "faiss uq 2 bit init  " << std::endl;
            this->ntotal = this->codes.size() / (this->d / 16); // n total: number of total vectors. should be codes.sz / 16. 
            parent->ntotal = this->ntotal;
            grand_parent->ntotal = this->ntotal;
        }
        faiss::FlatCodesDistanceComputer* get_FlatCodesDistanceComputer() const override {
            // std::cout << "faiss uq 2 bit distance computer  " << std::endl;
            return new knn_jni::faiss_wrapper::ADCFlatCodesDistanceComputer2Bit(
                (const uint8_t *) (this->codes.data()), this->code_size, this->d, this->metric_type,
                above_threshold_means, below_threshold_means
            );

        };


    };

    struct FaissIndexUQ4Bit : faiss::IndexFlatCodes {
        std::vector<float> above_threshold_means;
        std::vector<float> below_threshold_means;

        FaissIndexUQ4Bit(
            faiss::idx_t d, std::vector<uint8_t> codes, faiss::MetricType metric=faiss::METRIC_L2, std::vector<float> above_threshold_mean_vector= std::vector<float>(), std::vector<float> below_threshold_mean_vector= std::vector<float>()
        ) : IndexFlatCodes(d/2, d, metric){
            
            this->codes = codes; 
            this->code_size = (d/2);

            this->above_threshold_means = above_threshold_mean_vector;
            this->below_threshold_means = below_threshold_mean_vector;


        }

        void init(faiss::Index * parent, faiss::Index * grand_parent) {
            this->ntotal = this->codes.size() / (this->d /2);
            parent->ntotal = this->ntotal;   
            grand_parent->ntotal = this->ntotal;
        }
        faiss::FlatCodesDistanceComputer* get_FlatCodesDistanceComputer() const override {
            return new knn_jni::faiss_wrapper::ADCFlatCodesDistanceComputer4Bit(
                (const uint8_t *) (this->codes.data()), this->d/2, this->d, this->metric_type,above_threshold_means, below_threshold_means
            );

        }
    };
}
}
#endif //KNNPLUGIN_JNI_FAISSINDEXBQ_H
