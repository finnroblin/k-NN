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
            std::vector<std::vector<std::vector<float>>> lookup_table; // used in batched distances
            std::vector<float> coord_scores; // scores for each dimension
            float correction_amount; // used in batched distances

            std::vector<std::vector<float>> partitions;                
            std::vector<float> above_threshold_means;
            std::vector<float> below_threshold_means;

            std::vector<float> z;
            const int BATCH_SZ = 8; // probably needs to be powers of 8 since then it's grabbing a whole byte.
            // const int BATCH_SZ = 16;
            const int NUM_BATCHES = 0; // overridden in initializer list

            // TODO note: the dimension passed in from k-NN is the number of bits
            // so for 2 bit quant, and 128 dimension vectors, d = 256. 
            ADCFlatCodesDistanceComputer2Bit(const uint8_t * codes, size_t code_size, int d, faiss::MetricType metric_type = faiss::METRIC_L2,
                std::vector<float> above_threshold_means = std::vector<float>()
                , std::vector<float> below_threshold_means = std::vector<float>()
            )
            : FlatCodesDistanceComputer(codes, code_size), dimension(d), query(nullptr), metric_type(metric_type), NUM_BATCHES((d / 2) / BATCH_SZ ) {
                // std::cout << "faiss uq 2 bit distance computer ctor  " << std::endl;
                this->codes = codes;
                this->code_size = code_size;
                // TODO change to BIT_SZ
                this->dimension = d/2; // since the underlying index has dimension as 2 * vector_dim. 
                this->correction_amount = 0.0f; 
                this->above_threshold_means = above_threshold_means;
                
                this->below_threshold_means = below_threshold_means;
                // std::cout << " above thres first elt " << this->above_threshold_means[0] << std::endl;
                // std::cout << "this->correction_amoutn: " << std::to_string(this->correction_amount) << std::endl;

                // std::cout << "code size : " << this->code_size << " dimensions: " << this->dimension;
                this->partitions = std::vector<std::vector<float>>(
                    this->dimension, std::vector<float> (3 , 0.0f) // TODO magic constant
                ); // TODO change to BIT_SZ + 1

                for (int i = 0; i < this->dimension; ++i) {
                    float x = below_threshold_means[i];
                    float y = above_threshold_means[i]; 
                    partitions[i][0] = x;
                    partitions[i][1] = (x + y) / 2.0f;
                    partitions[i][2] = y;
                }

                // TODO change to BIT_SZ
                this->z = std::vector<float>(this->dimension*2, 0.0f ); // we want it to be d * 2

                // NUM_BATCHES = (this->dimension/NUM_BATCHES);
            }

            float distance_to_code_batched(const uint8_t * code) {
                float dist = 0.0f; 
                int num_bytes = (dimension/8);

                for (int dim_idx = 0; dim_idx < num_bytes; ++dim_idx) {
                    const unsigned char code_batch_first_bit = code[dim_idx];

                    const unsigned char code_batch_second_bit = code[dim_idx + num_bytes];

                    // TODO more efficient vector access pattern
                    dist += this->lookup_table[0][dim_idx][code_batch_first_bit];
                    dist += this->lookup_table[1][dim_idx][code_batch_second_bit];
                }

                return dist + correction_amount;
            }

            float distance_to_code_batched_old(const uint8_t* code) {
                float distance = 0.0f;
    
                // batch of 8 vector entries (16 bits)
                for (int batch_idx = 0; batch_idx < NUM_BATCHES; ++batch_idx) {
                    // We need to convert the packed 2-bit values to a base-3 index
                    int base3_index = 0;

                    const std::array<int, 8> powers = {1, 3, 9, 27, 81, 243, 729, 2187};
                    
                    // Get the bytes containing the first and second bits
                    uint8_t first_bits_byte = code[batch_idx];
                    uint8_t second_bits_byte = code[this->dimension/8 + batch_idx];
                    
                    for (int dim = 0; dim < BATCH_SZ; ++dim) {
                        // Extract the bits for this dimension
                        int first_bit = (first_bits_byte >> (7 - dim)) & 1;
                        int second_bit = (second_bits_byte >> (7 - dim)) & 1;
                        
                        // Convert bit pattern to value (0, 1, or 2)
                        int value = 0;
                        if (first_bit == 1 && second_bit == 0) {
                            value = 1;  // 10
                        } else if (first_bit == 1 && second_bit == 1) {
                            value = 2;  // 11
                        }
                        // 00 is value 0, 01 is invalid
                        
                        // Add to the base-3 index. Expensive operation, can probably be improved.
                        base3_index += value * powers[dim];
                        // power *= 3;
                    }
                    
                    // Look up the precomputed distance for this arrangement of vectors
                    // distance += this->lookup_table[batch_idx][base3_index];
                }
                
                return distance + this->correction_amount;            
            }

            

            virtual float distance_to_code(const uint8_t* code) override {
                return distance_to_code_batched(code);
                // return distance_to_code_unbatched(code);
            }

            float distance_to_code_unbatched(const uint8_t* code) {
                // tbe java bit packing code sets the first bit for all dimensions, then the second bit for all dimensions, etc.
                // compute p dot z
                float distance = 0.0f; 
                for (int code_byte_idx = 0; code_byte_idx < this->dimension / 8; ++code_byte_idx) {
                    uint8_t first_code_byte = code[code_byte_idx];
                    
                    // the matching bits for this document are at code_byte[code_bit], (code_byte+this->dimension/8)[code_bit]
                    uint8_t second_code_byte = code[(this->dimension / 8) + code_byte_idx];
                    for (int first_code_bit = 0; first_code_bit < 8; ++first_code_bit) {
                        
                        int first_code_entry = (first_code_byte >> (7 - first_code_bit)) & 1;
                        int second_code_entry = (second_code_byte >> (7 - first_code_bit)) & 1;

                        // TODO for 4 bit, change to code_byte_idx * 8 * BIT_SZ , first_code_bit * BIT_SZ.
                        int z_idx = code_byte_idx * 16 + first_code_bit * 2;
                        // std::cout << "before first z access " << std::endl;
                        float first_z_val = z[
                            z_idx
                        ];
                        float second_z_val = z[
                            z_idx + 1
                        ];

                        if (first_code_entry == 1 && second_code_entry == 0) {
                            distance += first_z_val;
                        }
                        else if (first_code_entry == 1 && second_code_entry == 1) {
                            distance += first_z_val + second_z_val;
                        } else if (first_code_entry == 0 && second_code_entry == 1) {
                            std::cout << "INVALID!!!" << std::endl;
                        }
                        // no contribution to distance for 0 case (handled via the correction_amount variable)
                    }
                }
                return distance + this->correction_amount;

            };

            void compute_z() {
                // reset z
                std::fill(z.begin(), z.end(), 0.0f);
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
            }

            virtual void set_query(const float* x) override {
                this->correction_amount = 0.0f;
                this->query = x;
                // compute z                
                compute_z();

                create_batched_lookup_table(); 
            };

            // batched optimization
            void compute_cord_scores() {
                const unsigned int num_batches = this->dimension / 8; 

                // const unsigned int num_possibilities_per_batch = 6561; // 3 ^ 8  = 6561

                for (int i  = 0; i < num_batches; ++i ) {
                    compute_per_batch_lookup_2_bit(i, 0, this->lookup_table[i][0]);
                    compute_per_batch_lookup_2_bit(i, 1, this->lookup_table[i][1]);
                }
            }

            void compute_per_batch_lookup_2_bit(int batch_idx, int which_bit, std::vector<float> & batch) {
                batch[0] = 0.0f; 
                for (int i = 0; i < 8; ++i) {
                    const unsigned int bit_masked = 1 << i;
                    // int batch_idx_offset = batch_idx * 8 * 2; // start location
                    // int first_or_second_bit_offset = 2 * which_bit; // which bit we're on (every odd entry corresponds to first bit in entry, every even entry to second bit in document entry)
                    // int code_bit_offset = (7 - i); // for building batch
                    // int code_bit_offset = i;
                    // int z_idx = batch_idx_offset + first_or_second_bit_offset + code_bit_offset;
                    int dim_idx = batch_idx * 8 + (7 - i);
                    int z_idx = dim_idx * 2 + which_bit;
                    const float bit_value = this->z[ 
                        z_idx
                    ];
                    for (unsigned int suffix = 0; suffix < bit_masked; ++suffix) {
                        batch[
                            bit_masked | suffix // range from 0 to 255
                        ] = batch[suffix] + bit_value;
                    }
                }
            }

            void compute_per_batch_lookup_2_bit_old(int batch_idx, std::vector<float> & batch) {
                // Powers of 3 for base-3 indexing
                const int pow3[] = {1, 3, 9, 27, 81, 243, 729, 2187};

                batch[0] = 0.0f;

                for (int vec_num = 0; vec_num < 8; ++vec_num) {
                    int z_idx = batch_idx * 2 * 8 + vec_num * 2;

                    const float value_for_first_bit= z[z_idx]; // first bit contribution 
                    const float value_for_second_bit = z[z_idx+1]; // second bit contribution

                    // Current base-3 position weight
                    const int weight = pow3[vec_num];

                    for (int prefix = 0; prefix < pow3[vec_num]; ++prefix) {
                        // Value 1: Only first bit set (10)
                        batch[prefix + weight] = batch[prefix] + value_for_first_bit;
                        
                        // Value 2: Both bits set (11)
                        batch[prefix + 2 * weight] = batch[prefix] + value_for_first_bit + value_for_second_bit;
                    }
                }
            }

            void create_batched_lookup_table() {    
                // batch size is 8 vectors, 3^8 possibilities per batch. 
                const unsigned int num_batches = this->dimension / 8;

                // Initialize lookup_table with the right dimensions
                // Each batch needs a table of size 3^8 = 6561
                // this->lookup_table.resize(NUM_BATCHES, std::vector<float>(6561, 0.0f));
                
                // for (int i = 0; i < NUM_BATCHES; ++i) {
                //     // compute_per_batch_lookup_2_bit(i, this->lookup_table[i]);
                // }
                // this->lookup_table.resize(
                //     num_batches,
                //     std::vector<std::vector<float>>(
                //         2, 
                //         std::vector<float> (256, 0.0f) 
                //     )
                // );
                this->lookup_table.resize(
                    2,
                    std::vector<std::vector<float>>(
                        num_batches,
                        std::vector<float>(256, 0.0f)
                    )
                );
                
                for (int i  = 0; i < num_batches; ++i ) {
                    // compute_per_batch_lookup_2_bit(i, 0, this->lookup_table[i][0]);
                    // compute_per_batch_lookup_2_bit(i, 1, this->lookup_table[i][1]);
                    compute_per_batch_lookup_2_bit(i, 0, this->lookup_table[0][i]);
                    compute_per_batch_lookup_2_bit(i, 1, this->lookup_table[1][i]);
                    
                }
            }

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

            std::vector<float> z;
            const int BATCH_SZ = 8; // probably needs to be powers of 8 since then it's grabbing a whole byte.
            // const int BATCH_SZ = 16;
            const int NUM_BATCHES = 0; // overridden in initializer list

            // TODO note: the dimension passed in from k-NN is the number of bits
            // so for 2 bit quant, and 128 dimension vectors, d = 256. 
 
            const int BIT_SZ = 4;

            ADCFlatCodesDistanceComputer4Bit(const uint8_t * codes, size_t code_size, int d, faiss::MetricType metric_type = faiss::METRIC_L2,
                std::vector<float> above_threshold_means= std::vector<float>(), std::vector<float> below_threshold_means = std::vector<float>()
            )

        : FlatCodesDistanceComputer(codes, code_size), dimension(d), query(nullptr), metric_type(metric_type), NUM_BATCHES((d / 2) / BATCH_SZ ) {
            // : FlatCodesDistanceComputer(codes, code_size), dimension(d), query(nullptr), metric_type(metric_type) {
                // std::cout << "faiss uq 4 bit distance computer ctor  " << std::endl;
                this->codes = codes;
                this->code_size = code_size;
                this->dimension = d/BIT_SZ;
                correction_amount = 0.0f; 
                
                this->above_threshold_means = above_threshold_means;
                this->below_threshold_means = below_threshold_means;

                this->partitions = std::vector<std::vector<float>>(
                    this->dimension, std::vector<float> ((BIT_SZ+1) , 0.0f) // d x (b + 1)
                );

                // Fill the partitions vector with interpolated values
// 
                // std::cout << "faiss uq 4 bit distance computer ctor  " << std::endl;

                for (int i = 0; i < this->dimension; ++i) {
                    // std::cout << "above/below trhes" << std::endl;
                    float x = below_threshold_means[i];
                    float y = above_threshold_means[i]; 
                    
                    // for (int j = 0; j <= BIT_SZ; ++j) {
                    //     // For each possible count of 1s (from 0 to BIT_SZ),
                    //     // linearly interpolate between x and y

                    //     // std::cout << "lin interpolation" << std::endl;
                    //     partitions[i][j] = (1.0 * (BIT_SZ - j) * x + j * y) / static_cast<float>(BIT_SZ);

                    //     // partitions[i][j] = (1.0 * (j) * x + (BIT_SZ - j) * y) / static_cast<float>(BIT_SZ);
                    // }
                    partitions[i][0] = x;
                    partitions[i][BIT_SZ] = y;
    
    // Non-linear spacing for intermediate partitions (gives more weight to center values)
                    for (int j = 1; j < BIT_SZ; ++j) {
                        float t = static_cast<float>(j) / BIT_SZ;
                        // Apply a slight non-linear transformation to better represent data distribution
                        // This gives more precision in the middle range where most vectors typically fall
                        // t = std::pow(t, 0.8); // Slight curve that emphasizes middle values
                        partitions[i][j] = (1.0f - t) * x + t * y;
                    }
                }
                // std::cout << "after interplt" << std::endl;

                this->z = std::vector<float>(this->dimension * BIT_SZ, 0.0f);
            }

            virtual float distance_to_code(const uint8_t* code) override {
                // return 0.0f;
                return distance_to_code_unbatched(code);
            };

            virtual void set_query(const float* x) override {
                this->correction_amount = 0.0f;
                this->query = x;
                // vcompute z
                // here we need to calculate the partitions. 
                compute_z();
            };

            float distance_to_code_unbatched(const uint8_t* code) {
                float distance = 0.0f;
                
                for (int code_byte_idx = 0; code_byte_idx < this->dimension / 8; ++code_byte_idx) {
                    // Get all 4 bit planes for this byte index
                    uint8_t bit0_byte = code[(0) * (this->dimension / 8) + code_byte_idx];
                    uint8_t bit1_byte = code[(1) * (this->dimension / 8) + code_byte_idx];
                    uint8_t bit2_byte = code[(2) * (this->dimension / 8) + code_byte_idx];
                    uint8_t bit3_byte = code[(3) * (this->dimension / 8) + code_byte_idx];
                    
                    for (int code_bit = 0; code_bit < 8; ++code_bit) {
                        // Calculate the base index in the z array for this dimension
                        int z_idx = code_byte_idx * 8 * BIT_SZ + code_bit * BIT_SZ;
                        
                        // Extract each bit at position code_bit
                        uint8_t bit0 = (bit0_byte >> (7 - code_bit)) & 1;
                        uint8_t bit1 = (bit1_byte >> (7 - code_bit)) & 1;
                        uint8_t bit2 = (bit2_byte >> (7 - code_bit)) & 1;
                        uint8_t bit3 = (bit3_byte >> (7 - code_bit)) & 1;
                        
                        // Process according to unary encoding logic - valid patterns are:
                        // 0000 = partition 0 (no distance added)
                        // 1000 = partition 1 (add z[0])
                        // 1100 = partition 2 (add z[0] + z[1])
                        // 1110 = partition 3 (add z[0] + z[1] + z[2])
                        // 1111 = partition 4 (add z[0] + z[1] + z[2] + z[3])
                        
                        // Using cascading conditionals for efficiency
                        if (bit0) {
                            distance += z[z_idx];
                            if (bit1) {
                                distance += z[z_idx + 1];
                                if (bit2) {
                                    distance += z[z_idx + 2];
                                    if (bit3) {
                                        distance += z[z_idx + 3];
                                    }
                                }
                            }
                        }
                    }
                }
                
                return distance + this->correction_amount;
            }
            float distance_to_code_unbatched_old(const uint8_t* code) {
                // tbe java bit packing code sets the first bit for all dimensions, then the second bit for all dimensions, etc.
                // compute p dot z
                float distance = 0.0f;
                
                // const int num_bits = 4; // Now using 4 bits instead of 2
                
                for (int code_byte_idx = 0; code_byte_idx < this->dimension / 8; ++code_byte_idx) {
                    // Get all code bytes for this byte index
                    uint8_t code_bytes[BIT_SZ];
                    for (int bit_pos = 0; bit_pos <BIT_SZ ; ++bit_pos) {
                        code_bytes[bit_pos] = code[bit_pos * (this->dimension / 8) + code_byte_idx];
                    }


                    // TODO make this less complicated
                    
                    for (int code_bit = 0; code_bit < 8; ++code_bit) {
                        // Calculate the base index in the z array
                        int z_idx = code_byte_idx * 8 *BIT_SZ  + code_bit *BIT_SZ ;
                        
                        // Extract bits for this position
                        int bit_values[BIT_SZ];
                        for (int bit_pos = 0; bit_pos < BIT_SZ ; ++bit_pos) {
                            bit_values[bit_pos] = (code_bytes[bit_pos] >> (7 - code_bit)) & 1;
                        }
                        
                        // Check if the bit pattern is valid (unary encoding requires 1s followed by 0s)
                        bool valid = true;
                        for (int i = 1; i <BIT_SZ ; ++i) {
                            if (bit_values[i-1] == 0 && bit_values[i] == 1) {
                                std::cout << "INVALID!!!" << std::endl;
                                valid = false;
                                break;
                            }
                        }
                        // TODO this needs to change and not be so slow with the array accesses. Probably templating is the move here.
                        // if (valid) {
                            // Calculate the contribution to the distance
                            int num_ones = 0;
                            for (int i = 0; i < BIT_SZ; ++i) {
                                if (bit_values[i] == 1) {
                                    distance += z[z_idx + i];
                                    // num_ones++;
                                }
                            }
                            // For debugging
                            // if (num_ones > 0) {
                            //     std::cout << "Found " << num_ones << " ones, adding values from z[" << z_idx << "] to z[" << (z_idx+num_ones-1) << "]" << std::endl;
                            // }
                        // }
                        // Count the number of leading 1s (this is the partition index)
                        // int num_ones = 0;
                        // for (int i = 0; i < BIT_SZ; ++i) {
                        //     if (bit_values[i] == 1) {
                        //         num_ones++;
                        //     } else {
                        //         break; // In unary encoding, once we hit a 0, all remaining bits should be 0
                        //     }
                        // }

                        // // Add z values for this partition (all values up to num_ones)
                        // for (int i = 0; i < num_ones; ++i) {
                        //     distance += z[z_idx + i];
                        // }


                    }
                }
                return distance + this->correction_amount;
            }
            void compute_z() {
                std::fill(z.begin(), z.end(), 0.0f);
                this->correction_amount = 0.0f;
                
                for (int dim = 0; dim < this->dimension; ++dim) {
                    float query_value = this->query[dim];
                    
                    // Pre-calculate all squared distances
                    std::vector<float> squared_distances(BIT_SZ + 1);
                    for (int p = 0; p <= BIT_SZ; p++) {
                        float diff = query_value - this->partitions[dim][p];
                        squared_distances[p] = diff * diff;
                    }
                    
                    // Base correction amount
                    this->correction_amount += squared_distances[0];
                    float accumulator = squared_distances[0];
                    
                    // Calculate incremental distances with better numerical precision
                    for (int p = 1; p <= BIT_SZ; p++) {
                        float increment = squared_distances[p] - accumulator;
                        
                        // Handle potential numerical instability
                        if (increment < 0 && std::abs(increment) < 1e-6) {
                            increment = 0;
                        }
                        
                        z[dim * BIT_SZ + (p-1)] = increment;
                        accumulator += increment;
                    }
                }
            }
            
            
            void compute_z_old() {
                // Reset z vector and correction amount
                std::fill(z.begin(), z.end(), 0.0f);
                this->correction_amount = 0.0f;
                
                for (int dim = 0; dim < this->dimension; ++dim) {
                    float query_value = this->query[dim];
                    
                    // Calculate the squared distance to the first partition value (partition 0)
                    // This becomes our baseline "correction amount" - the minimum distance contribution
                    float base_distance = (query_value - this->partitions[dim][0]) * (query_value - this->partitions[dim][0]);
                    float accumulator = base_distance;
                    this->correction_amount += base_distance;
                    
                    // For each remaining partition (1 through 4 for 4-bit encoding)
                    // Calculate the incremental distance contribution of this partition
                    for (int p = 1; p <= BIT_SZ; p++) {
                        // Compute squared distance to this partition value
                        float partition_value = this->partitions[dim][p];
                        float squared_dist = (query_value - partition_value) * (query_value - partition_value);
                        
                        // Calculate the incremental distance (difference from what we've accumulated so far)
                        float increment = squared_dist - accumulator;
                        
                        // Store this increment in the z array
                        z[dim * BIT_SZ + (p-1)] = increment;
                        
                        // Update accumulator for next partition
                        accumulator += increment;
                    }
                }
                
                // At this point:
                // - correction_amount is the minimum distance (partition 0 for all dimensions)
                // - z contains the incremental distances for partitions 1-4
                // - When computing distance, we'll add z[i] values based on the unary encoding pattern
            }
                        
            // void compute_z() {
            //     // reset z
            //     std::fill(z.begin(), z.end(), 0.0f);
            //     // this->correction_amount = 0.0f;
                
            //     for (int i = 0; i < this->dimension; ++i) {
            //         float c = this->query[i];
            //         float first_partition_distance = (c - this->partitions[i][0]) * (c - this->partitions[i][0]);
                    
            //         float accumulator = first_partition_distance;
            //         this->correction_amount += first_partition_distance;
                    
            //         // Now handle 5 partitions (for 4 bits) instead of 3 partitions (for 2 bits)
            //         for (int partition_idx = 1; partition_idx < BIT_SZ + 1; partition_idx++) {
            //             // std::cout << "partition idx" << std::endl;
            //             float m = this->partitions[i][partition_idx];
                        
            //             float v = (c - m) * (c - m);
            //             float d = v - accumulator;
            //             // std::cout << "z access " << std::endl;
            //             z[i * BIT_SZ + partition_idx - 1] = d;
            //             accumulator += d;
            //         }
             
            //     }

            //     // std::cout << "compute z done" << std::endl;
            // }
                        

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
                // std::cout << "customer flat codes called " << std::endl;
            }

            float distance_to_code_l2_unbatched(const uint8_t* code) {
                // L2 distance
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

                   // L2
                   float dim_score = (code_translated - query[i]) * (code_translated - query[i]);

                   score += dim_score;
               }
               return score;       
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
                // return distance_to_code_l2_unbatched(code);
    
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
                    std::cout << "computing with L2 metric" << std::endl;
                    compute_cord_scores_l2(); // todo make this templated based on space type.
                } else if (this->metric_type == faiss::METRIC_INNER_PRODUCT) {
                    std::cout << "computer with IP metric" << std::endl;
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
                
                for (int i = 0 ; i < this->dimension; ++i) {
                    
                    // query: 1 -2 4 5
                    // codes: 0  1 0 1
                    // inner_prod(query, codes) = 0 * 1 + -2 * 1 + 4 * 0 + 5 * 1
                    this->coord_scores[i] = query[i];
                    // correction_amount += x * x;
                }
            }



            virtual void set_query(const float* x) override {
                this->correction_amount = 0.0f;
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
            FaissIndexBQ(
                faiss::idx_t d,
                std::vector<uint8_t> * codes_ptr,
                faiss::MetricType metric=faiss::METRIC_L2
            ) : IndexFlatCodes(d/8, d, metric) {
                std::cout << " in the proper constructor, hopefully it works!" << std::endl;
                this->code_size = (d/ 8);
            }
            
            
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
        std::vector<uint8_t> * codes_ptr;
        FaissIndexUQ2Bit(
            faiss::idx_t d, std::vector<uint8_t> * codes_ptr, faiss::MetricType metric=faiss::METRIC_L2, std::vector<float> above_threshold_mean_vector = std::vector<float>(), std::vector<float> below_threshold_mean_vector= std::vector<float>()
        ) : IndexFlatCodes(d/8, d, metric){
            // std::cout << "faiss uq 2 bit ctor changed , dimension " << d << "."  << std::endl;
            // std::cout << " code ptr " << static_cast<char>((*codes_ptr)[0]) << std::endl;
            // this->codes = codes; 
            // std::cout << " after code ptr access " << std::endl;
            this->codes_ptr = codes_ptr;
            this->code_size = (d/ 8);
            this->above_threshold_means = above_threshold_mean_vector;
            this->below_threshold_means = below_threshold_mean_vector;
        }

        // FaissIndexUQ2Bit(
        //     faiss::idx_t d, std::vector<uint8_t> codes, faiss::MetricType metric=faiss::METRIC_L2, std::vector<float> above_threshold_mean_vector = std::vector<float>(), std::vector<float> below_threshold_mean_vector= std::vector<float>()
        // ) : IndexFlatCodes(d/8, d, metric){
        //     // std::cout << "faiss uq 2 bit ctor , dimension " << d << "."  << std::endl;
        //     this->codes = codes; 
        //     this->code_size = (d/ 8);
        //     this->above_threshold_means = above_threshold_mean_vector;
        //     this->below_threshold_means = below_threshold_mean_vector;
        // }

        void init(faiss::Index * parent, faiss::Index * grand_parent) {
            // std::cout << "faiss uq 2 bit init  " << std::endl;
            this->ntotal = this->codes_ptr->size() / (this->d / 16); // n total: number of total vectors. should be codes.sz / 16. 
            parent->ntotal = this->ntotal;
            grand_parent->ntotal = this->ntotal;
        }
        faiss::FlatCodesDistanceComputer* get_FlatCodesDistanceComputer() const override {
            // std::cout << "faiss uq 2 bit distance computer  " << std::endl;
            // // return new knn_jni::faiss_wrapper::ADCFlatCodesDistanceComputer2Bit(
            //     (const uint8_t *) (this->codes.data()), this->code_size, this->d, this->metric_type,
            //     above_threshold_means, below_threshold_means
            // );
            return new knn_jni::faiss_wrapper::ADCFlatCodesDistanceComputer2Bit(
                (this->codes_ptr->data()), this->code_size, this->d, this->metric_type,
                above_threshold_means, below_threshold_means
            );

        };


    };

    struct FaissIndexUQ4Bit : faiss::IndexFlatCodes {
        std::vector<float> above_threshold_means;
        std::vector<float> below_threshold_means;
        const int BIT_SZ = 4;

        FaissIndexUQ4Bit(
            faiss::idx_t d, std::vector<uint8_t> codes, faiss::MetricType metric=faiss::METRIC_L2, std::vector<float> above_threshold_mean_vector= std::vector<float>(), std::vector<float> below_threshold_mean_vector= std::vector<float>()
        ) : IndexFlatCodes(d/8, d, metric){
            std::cout << " in uq 4 bit ctor , dim : "  << d << std::endl;
            this->codes = codes; 
            this->code_size = (d/8);

            this->above_threshold_means = above_threshold_mean_vector;
            this->below_threshold_means = below_threshold_mean_vector;


        }

        void init(faiss::Index * parent, faiss::Index * grand_parent) {
            this->ntotal = this->codes.size() / (this->d / (8 * BIT_SZ));
            parent->ntotal = this->ntotal;   
            grand_parent->ntotal = this->ntotal;
        }
        faiss::FlatCodesDistanceComputer* get_FlatCodesDistanceComputer() const override {
            return new knn_jni::faiss_wrapper::ADCFlatCodesDistanceComputer4Bit(
                (const uint8_t *) (this->codes.data()), this->code_size, this->d, this->metric_type,above_threshold_means, below_threshold_means
            );

        }
    };
}
}
#endif //KNNPLUGIN_JNI_FAISSINDEXBQ_H
