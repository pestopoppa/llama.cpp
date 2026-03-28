#include "llama-hadamard.h"

#include <cmath>
#include <cstring>

// Standard iterative butterfly algorithm for Walsh-Hadamard Transform.
// For n=128 (our KV head dimension): 7 stages × 64 butterflies = 448 add/sub ops.
// Normalized by 1/sqrt(n) so H·H = I (self-inverse / orthogonal).
void fwht_inplace(float * data, int n) {
    for (int len = 1; len < n; len <<= 1) {
        for (int i = 0; i < n; i += len << 1) {
            for (int j = 0; j < len; j++) {
                float u = data[i + j];
                float v = data[i + j + len];
                data[i + j]       = u + v;
                data[i + j + len] = u - v;
            }
        }
    }
    float scale = 1.0f / sqrtf((float)n);
    for (int i = 0; i < n; i++) {
        data[i] *= scale;
    }
}

// ggml_map_custom1 callback.
// Applies WHT independently to each row (dim0 slice) of the tensor.
// src shape: [d, ...] where d is the WHT dimension (must be power of 2).
// Rows are partitioned across threads for parallelism.
void ggml_hadamard_custom_op(
        struct ggml_tensor * dst,
        const struct ggml_tensor * src,
        int ith, int nth, void * userdata) {
    (void)userdata;

    const int d = (int)src->ne[0];  // WHT dimension (e.g. 128)
    const int nrows = (int)ggml_nrows(src);

    // Partition rows across threads
    const int rows_per_thread = (nrows + nth - 1) / nth;
    const int row_start = ith * rows_per_thread;
    const int row_end   = row_start + rows_per_thread < nrows ? row_start + rows_per_thread : nrows;

    for (int row = row_start; row < row_end; row++) {
        const float * src_row = (const float *)((const char *)src->data + row * src->nb[1]);
        float       * dst_row = (float       *)((char       *)dst->data + row * dst->nb[1]);

        // Copy src to dst if not in-place
        if (src_row != dst_row) {
            memcpy(dst_row, src_row, d * sizeof(float));
        }

        fwht_inplace(dst_row, d);
    }
}
