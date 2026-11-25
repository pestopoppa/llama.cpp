#include "concat.cuh"
#include "knee.cuh"

#include <stdint.h>

// contiguous kernels
template <typename T, int dim>
static __global__ void __launch_bounds__(CUDA_CONCAT_BLOCK_SIZE) concat_cont(const T * x,
                                                                             const T * y,
                                                                             T *       dst,
                                                                             int64_t   ne00,
                                                                             int64_t   ne01,
                                                                             int64_t   ne02,
                                                                             int64_t   ne0,
                                                                             int64_t   ne1,
                                                                             int64_t   ne2) {
    static_assert(dim >= 0 && dim <= 2, "dim must be in [0, 2]");

    const int64_t n = ne0 * ne1 * ne2;

    ggml_cuda_pdl_sync();
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t) blockDim.x * gridDim.x) {
        if constexpr (dim == 0) {
            const int64_t row = i / ne0;
            const int64_t i0  = i - row * ne0;

            if (i0 < ne00) {
                dst[i] = x[row * ne00 + i0];
            } else {
                dst[i] = y[row * (ne0 - ne00) + (i0 - ne00)];
            }
        } else if constexpr (dim == 1) {
            const int64_t dst_plane  = ne0 * ne1;
            const int64_t src0_plane = ne0 * ne01;
            const int64_t src1_plane = dst_plane - src0_plane;
            const int64_t i2         = i / dst_plane;
            const int64_t i01        = i - i2 * dst_plane;

            if (i01 < src0_plane) {
                dst[i] = x[i2 * src0_plane + i01];
            } else {
                dst[i] = y[i2 * src1_plane + (i01 - src0_plane)];
            }
        } else {
            const int64_t src0_size = ne0 * ne1 * ne02;

            if (i < src0_size) {
                dst[i] = x[i];
            } else {
                dst[i] = y[i - src0_size];
            }
        }
    }
}

template <typename T>
static void concat_cont_cuda(const T * x,
                             const T * y,
                             T *       dst,
                             int64_t   ne00,
                             int64_t   ne01,
                             int64_t   ne02,
                             int64_t   ne0,
                             int64_t   ne1,
                             int64_t   ne2,
                             int       dim,
                             cudaStream_t stream) {
    const int64_t n          = ne0 * ne1 * ne2;
    const int     num_blocks = (n + CUDA_CONCAT_BLOCK_SIZE - 1) / CUDA_CONCAT_BLOCK_SIZE;

    if (dim == 0) {
        const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(num_blocks, CUDA_CONCAT_BLOCK_SIZE, 0, stream);
        ggml_cuda_kernel_launch(concat_cont<T, 0>, launch_params, x, y, dst, ne00, ne01, ne02, ne0, ne1, ne2);
        return;
    }
    if (dim == 1) {
        concat_cont<T, 1><<<num_blocks, CUDA_CONCAT_BLOCK_SIZE, 0, stream>>>(x, y, dst, ne00, ne01, ne02, ne0, ne1, ne2);
        return;
    }
    concat_cont<T, 2><<<num_blocks, CUDA_CONCAT_BLOCK_SIZE, 0, stream>>>(x, y, dst, ne00, ne01, ne02, ne0, ne1, ne2);
}

// non-contiguous kernel (slow)
template <typename T, int dim>
static __global__ void __launch_bounds__(CUDA_CONCAT_BLOCK_SIZE)
    concat_non_cont(
        const char * src0,
        const char * src1,
              char * dst,
           int64_t   ne00,
           int64_t   ne01,
           int64_t   ne02,
           int64_t   ne03,
          uint64_t   nb00,
          uint64_t   nb01,
          uint64_t   nb02,
          uint64_t   nb03,
           int64_t /*ne10*/,
           int64_t /*ne11*/,
           int64_t /*ne12*/,
           int64_t /*ne13*/,
          uint64_t   nb10,
          uint64_t   nb11,
          uint64_t   nb12,
          uint64_t   nb13,
           int64_t   ne0,
           int64_t /*ne1*/,
           int64_t /*ne2*/,
           int64_t /*ne3*/,
          uint64_t   nb0,
          uint64_t   nb1,
          uint64_t   nb2,
          uint64_t   nb3) {
    static_assert(dim >= 0 && dim <= 3, "dim must be in [0, 3]");

    const int64_t i3 = blockIdx.z;
    const int64_t i2 = blockIdx.y;
    const int64_t i1 = blockIdx.x;

    const T * x;

    for (int64_t i0 = threadIdx.x; i0 < ne0; i0 += blockDim.x) {
        if (i0 < ne00 && i1 < ne01 && i2 < ne02 && i3 < ne03) {
            x = (const T *)(src0 + i3*nb03 + i2*nb02 + i1*nb01 + i0*nb00);
        } else {
            if constexpr (dim == 0) {
                x = (const T *)(src1 + i3*nb13 + i2*nb12 + i1*nb11 + (i0 - ne00)*nb10);
            } else if constexpr (dim == 1) {
                x = (const T *)(src1 + i3*nb13 + i2*nb12 + (i1 - ne01)*nb11 + i0*nb10);
            } else if constexpr (dim == 2) {
                x = (const T *)(src1 + i3*nb13 + (i2 - ne02)*nb12 + i1*nb11 + i0*nb10);
            } else if constexpr (dim == 3) {
                x = (const T *)(src1 + (i3 - ne03)*nb13 + i2*nb12 + i1*nb11 + i0*nb10);
            }
        }

        T * y = (T *)(dst + i3*nb3 + i2*nb2 + i1*nb1 + i0*nb0);

        *y = *x;
    }
}


// GGML_HIP_KNEE_CONCAT=1 (HBM-knee batch, default off): the non-contiguous concat as ONE flat index over dst, each
// workgroup owning a contiguous [i0, i1) range, grid sized by ggml_cuda_knee_grid (4 elements per lane in flight). The
// stock kernel launches one 256-thread WG per dst row, so a short-row concat (GDN conv state: ne0 = 3 + n_tokens) runs
// 256-thread WGs with ~11 active lanes. Same element mapping, same values (pure copy), any dim.
#define CONCAT_KNEE_UNROLL 4
template <typename T, int dim>
static __global__ void __launch_bounds__(CUDA_CONCAT_BLOCK_SIZE)
    concat_non_cont_knee(const char * src0, const char * src1, char * dst,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const uint64_t nb00, const uint64_t nb01, const uint64_t nb02, const uint64_t nb03,
        const uint64_t nb10, const uint64_t nb11, const uint64_t nb12, const uint64_t nb13,
        const int64_t ne0, const int64_t ne1, const int64_t ne2, const int64_t ne3,
        const uint64_t nb0, const uint64_t nb1, const uint64_t nb2, const uint64_t nb3) {
    static_assert(dim >= 0 && dim <= 3, "dim must be in [0, 3]");
    const int64_t n = ne0 * ne1 * ne2 * ne3;
    int64_t r0, r1;
    ggml_cuda_knee_range(n, blockIdx.x, gridDim.x, r0, r1);
    for (int64_t base = r0 + threadIdx.x; base < r1; base += (int64_t) blockDim.x * CONCAT_KNEE_UNROLL) {
        T v[CONCAT_KNEE_UNROLL];
        char * y[CONCAT_KNEE_UNROLL];
#pragma unroll
        for (int u = 0; u < CONCAT_KNEE_UNROLL; ++u) {
            const int64_t i = base + (int64_t) u * blockDim.x;
            y[u] = nullptr;
            if (i >= r1) {
                continue;
            }
            int64_t t = i;
            const int64_t i0 = t % ne0; t /= ne0;
            const int64_t i1 = t % ne1; t /= ne1;
            const int64_t i2 = t % ne2;
            const int64_t i3 = t / ne2;
            const char * x;
            if (i0 < ne00 && i1 < ne01 && i2 < ne02 && i3 < ne03) {
                x = src0 + i3*nb03 + i2*nb02 + i1*nb01 + i0*nb00;
            } else if constexpr (dim == 0) {
                x = src1 + i3*nb13 + i2*nb12 + i1*nb11 + (i0 - ne00)*nb10;
            } else if constexpr (dim == 1) {
                x = src1 + i3*nb13 + i2*nb12 + (i1 - ne01)*nb11 + i0*nb10;
            } else if constexpr (dim == 2) {
                x = src1 + i3*nb13 + (i2 - ne02)*nb12 + i1*nb11 + i0*nb10;
            } else {
                x = src1 + (i3 - ne03)*nb13 + i2*nb12 + i1*nb11 + i0*nb10;
            }
            v[u] = *(const T *) x;   // all loads of the unrolled group issue before any store
            y[u] = dst + i3*nb3 + i2*nb2 + i1*nb1 + i0*nb0;
        }
#pragma unroll
        for (int u = 0; u < CONCAT_KNEE_UNROLL; ++u) {
            if (y[u]) {
                *(T *) y[u] = v[u];
            }
        }
    }
}

template <typename T>
static void concat_cuda(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, int dim, cudaStream_t stream) {
    if (dim != 3 && ggml_is_contiguous_to_3(src0) && ggml_is_contiguous_to_3(src1)) {
        const T * src0_d = (const T *) src0->data;
        const T * src1_d = (const T *) src1->data;
        T *       dst_d  = (T *) dst->data;

        for (int64_t i3 = 0; i3 < dst->ne[3]; i3++) {
            concat_cont_cuda(
                    src0_d + i3*(src0->nb[3] / sizeof(T)),
                    src1_d + i3*(src1->nb[3] / sizeof(T)),
                    dst_d  + i3*( dst->nb[3] / sizeof(T)),
                    ggml_row_size(src0->type, src0->ne[0])/sizeof(T), src0->ne[1], src0->ne[2],
                    ggml_row_size(dst->type, dst->ne[0])/sizeof(T),  dst->ne[1],  dst->ne[2], dim, stream);
        }
    } else if (dim == 3 && ggml_is_contiguous(src0) && ggml_is_contiguous(src1)) {
        const size_t size0 = ggml_nbytes(src0);
        const size_t size1 = ggml_nbytes(src1);

        CUDA_CHECK(cudaMemcpyAsync((char *) dst->data,         src0->data, size0, cudaMemcpyDeviceToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync((char *) dst->data + size0, src1->data, size1, cudaMemcpyDeviceToDevice, stream));
    } else {
        GGML_ASSERT(!ggml_is_quantized(src0->type));

        static const bool knee = ggml_cuda_knee_env("GGML_HIP_KNEE_CONCAT");
        if (knee) {
            static bool logged = false;
            if (!logged) {
                logged = true;
                GGML_LOG_INFO("ggml_cuda_concat: GGML_HIP_KNEE_CONCAT knee flat-range non-cont concat ran (knee %d KB/CU)\n",
                              ggml_cuda_knee_kb());
            }
            int dev = 0;
            CUDA_CHECK(cudaGetDevice(&dev));
            const int64_t n      = ggml_nelements(dst);
            const int64_t wave_b = (int64_t) ggml_cuda_info().devices[dev].warp_size * CONCAT_KNEE_UNROLL * (int64_t) sizeof(T);
            const int64_t per_wg = (int64_t) CUDA_CONCAT_BLOCK_SIZE * CONCAT_KNEE_UNROLL;
            const int64_t nwg    = ggml_cuda_knee_grid(dev, CUDA_CONCAT_BLOCK_SIZE, wave_b, (n + per_wg - 1) / per_wg);
            auto launch_knee = [&](auto dim) {
                concat_non_cont_knee<T, dim><<<(unsigned) nwg, CUDA_CONCAT_BLOCK_SIZE, 0, stream>>>(
                    (const char *) src0->data, (const char *) src1->data, (char *) dst->data,
                    src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
                    src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
                    src1->nb[0], src1->nb[1], src1->nb[2], src1->nb[3],
                    dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
                    dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3]);
            };
            switch (dim) {
                case 0: launch_knee(std::integral_constant<int, 0>{}); break;
                case 1: launch_knee(std::integral_constant<int, 1>{}); break;
                case 2: launch_knee(std::integral_constant<int, 2>{}); break;
                case 3: launch_knee(std::integral_constant<int, 3>{}); break;
                default: GGML_ABORT("Invalid dim: %d", dim);
            }
            CUDA_CHECK(cudaGetLastError());
            return;
        }

        dim3 grid_dim(dst->ne[1], dst->ne[2], dst->ne[3]);
        auto launch_kernel = [&](auto dim) {
            concat_non_cont<T, dim><<<grid_dim, CUDA_CONCAT_BLOCK_SIZE, 0, stream>>>(
                (const char *) src0->data, (const char *) src1->data, (char *) dst->data,
                src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
                src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
                src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3],
                src1->nb[0], src1->nb[1], src1->nb[2], src1->nb[3],
                dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
                dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3]);
        };
        switch (dim) {
            case 0:
                launch_kernel(std::integral_constant<int, 0>{});
                break;
            case 1:
                launch_kernel(std::integral_constant<int, 1>{});
                break;
            case 2:
                launch_kernel(std::integral_constant<int, 2>{});
                break;
            case 3:
                launch_kernel(std::integral_constant<int, 3>{});
                break;
            default:
                GGML_ABORT("Invalid dim: %d", dim);
                break;
        }
    }
}

void ggml_cuda_op_concat(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    cudaStream_t stream = ctx.stream();

    const int32_t dim = ((int32_t *) dst->op_params)[0];

    GGML_ASSERT(src0->type == src1->type);
    GGML_ASSERT(dst->type  == src0->type);

    if (ggml_is_quantized(src0->type)) {
        if (dim == 3) {
            GGML_ASSERT(ggml_is_contiguous(src0));
            GGML_ASSERT(ggml_is_contiguous(src1));
        } else {
            GGML_ASSERT(ggml_is_contiguous_to_3(src0));
            GGML_ASSERT(ggml_is_contiguous_to_3(src1));
        }
        GGML_ASSERT(src0->ne[0] % ggml_blck_size(src0->type) == 0);
        GGML_ASSERT(src1->ne[0] % ggml_blck_size(src1->type) == 0);

        // if first 3 dimensions are contiguous and ne[0] is multiple of the block size we can concat both tensors as byte tensors
        concat_cuda<uint8_t>(src0, src1, dst, dim, stream);
    } else {
        GGML_ASSERT(ggml_blck_size(src0->type) == 1);

        switch (ggml_type_size(src0->type)) {
            case 1:
                concat_cuda<uint8_t>(src0, src1, dst, dim, stream);
                break;
            case 2:
                concat_cuda<uint16_t>(src0, src1, dst, dim, stream);
                break;
            case 4:
                concat_cuda<uint32_t>(src0, src1, dst, dim, stream);
                break;
            case 8:
                concat_cuda<uint64_t>(src0, src1, dst, dim, stream);
                break;
            default:
                GGML_ABORT("Unsupported type size: %zu", ggml_type_size(src0->type));
                break;
        }
    }
}
