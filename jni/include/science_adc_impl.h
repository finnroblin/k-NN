float set_coord_scores(const vector q, float *buf) {
    /* Compute the bitwise lookup table for `q` and store it in `buf`.
     * The squared distance to 0 is q.data[i]^2 and the squared distance to 1 is 1 - 2 q.data[i] + q.data[i]^2.
     * So the difference is 1 - 2 * q.data[i], and we can later add q.data[i]^2 to correct things.
     * const vector q: The query vector for which we produce the lookup table.
     * float *buf: The buffer in which the lookup table is stored. 
     * return float: The correction term  we need to add to the final distance calculation if we use this table. */

    float correction = 0.0f;
    const float *data = q.data;

    for (unsigned int i = 0; i < q.dimension; i++) {
        const float x = data[i];
        buf[i] = 1 - 2 * x;
        correction += x * x;
    }
    return correction;
}

void compute_all_sums(const float *orig_values, const size_t batch_size, float *out) {
    /**
     * Given an array 'orig_values' of length 'batch_size', compute the sums of all 2^batch_size subsets of orig_values.
     * const float *orig_values: An array of floats whose sums we will be computing
     * const size_t batch_size: The length of orig_values
     * float *out: The output buffer of length 2^batch_size. The value at each location i is the sum of orig_values
     *             masked by i. For example, out[0b11000010] = orig_values[1] + orig_values[6] + orig_values[7].
     */

    // Initialize empty sum to 0.0.
    out[0] = 0.f;

    /* Build the table up to bit b by duplicating the table already computed for bits 0..b-1,
     * plus the value at that new bit. */
    for (unsigned int bit = 0; bit < batch_size; bit++) {
        const unsigned int bit_masked = 1 << bit;
        const float bit_value = orig_values[batch_size - bit - 1];

        for (unsigned int suffix = 0; suffix < bit_masked; suffix++) {
            out[bit_masked | suffix] = out[suffix] + bit_value;
        }
    }
}

inline float batched_distance(const unsigned char *quantized_vec,
                              const float batched_sums[][256],
                              const unsigned int num_batches) {
    /**
     * Compute the ADC distance to the quantized vector 'quantized_vec' by adding the values from batched_sums as
     * masked by quantized_vec.
     * const unsigned char *quantized_vec: The quantized representation of the vector we compute the distance to.
     * const float batched_sums[][256]: An array of all possible values that a group of 8 bits can contribute.
     * const unsigned int num_batches: The length of batched_sums.
     * returns float: The distance to quantized_vec.
     */
    float dist = 0.0f;

    for (unsigned int batch_idx = 0; batch_idx < num_batches; batch_idx++) {
        const unsigned char batch = quantized_vec[batch_idx]; // read in next batch of bits
        dist += batched_sums[batch_idx][batch]; // update distance by the value in the batched look table
    }
    return dist;
}

nn_output asymmetric_get_nn_batched(const dataset_b D, const vector query) {
    /**
     * Finds the vector closest to 'query' among the binary-quantized dataset D.
     * const dataset_b D: The binary quantized dataset struct.
     * const vector query: An un-quantized query vector.
     * returns nn_output: The index of the nearest neighbor in D, plus its squared distance.
     */
    const unsigned int num_records = D.num_records, dimension = D.dimension;

    // initialize and precompute the lookup table for each bit individually
    float coord_scores[dimension];
    float correction = set_coord_scores(query, coord_scores);

    // initialize the lookup table for batches of bits
    const size_t batch_size = 8; // how many bits are in each batch
    const unsigned int num_batches = D.dimension / batch_size; // how many batches per vector
    float batch_scores[num_batches][1 << batch_size]; // stores the precomputed lookup table for each batch of bits

    // precompute the lookup table for each batch
    for (unsigned int batch_idx = 0; batch_idx < num_batches; batch_idx++)
        compute_all_sums(&coord_scores[batch_size * batch_idx], batch_size, batch_scores[batch_idx]);

    // find the nearest neighbor
    float dist, smallest_dist = FLT_MAX;
    unsigned int smallest_idx;

    for (unsigned int vector_idx = 0; vector_idx < num_records; vector_idx++) {
        vector_b vec = binary_get_vector(D, vector_idx);
        dist = batched_distance(vec.data, batch_scores, num_batches);

        if (dist < smallest_dist) {
            smallest_dist = dist;
            smallest_idx = vector_idx;
        }
    }

    return (nn_output){.index = smallest_idx, .distance = smallest_dist + correction};