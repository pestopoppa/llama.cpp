#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

    // the compute plan that needs to be prepared for ggml_graph_compute()
    // since https://github.com/ggml-org/ggml/issues/287
    struct ggml_cplan {
        size_t    work_size; // size of work buffer, calculated by `ggml_graph_plan()`
        uint8_t * work_data; // work buffer, to be allocated by caller before calling to `ggml_graph_compute()`

        int n_threads;
        struct ggml_threadpool * threadpool;

        // abort ggml_graph_compute when true
        ggml_abort_callback abort_callback;
        void *              abort_callback_data;

        // use only reference implementations
        bool use_ref;
    };

    // numa strategies
    enum ggml_numa_strategy {
        GGML_NUMA_STRATEGY_DISABLED   = 0,
        GGML_NUMA_STRATEGY_DISTRIBUTE = 1,
        GGML_NUMA_STRATEGY_ISOLATE    = 2,
        GGML_NUMA_STRATEGY_NUMACTL    = 3,
        GGML_NUMA_STRATEGY_MIRROR     = 4,
        GGML_NUMA_STRATEGY_COUNT
    };

    GGML_BACKEND_API void    ggml_numa_init(enum ggml_numa_strategy numa); // call once for better performance on NUMA systems
    GGML_BACKEND_API bool    ggml_is_numa(void); // true if init detected that system has >1 NUMA node

    GGML_BACKEND_API struct ggml_tensor * ggml_new_i32(struct ggml_context * ctx, int32_t value);
    GGML_BACKEND_API struct ggml_tensor * ggml_new_f32(struct ggml_context * ctx, float value);

    GGML_BACKEND_API struct ggml_tensor * ggml_set_i32 (struct ggml_tensor * tensor, int32_t value);
    GGML_BACKEND_API struct ggml_tensor * ggml_set_f32 (struct ggml_tensor * tensor, float value);

    GGML_BACKEND_API int32_t ggml_get_i32_1d(const struct ggml_tensor * tensor, int i);
    GGML_BACKEND_API void    ggml_set_i32_1d(const struct ggml_tensor * tensor, int i, int32_t value);

    GGML_BACKEND_API int32_t ggml_get_i32_nd(const struct ggml_tensor * tensor, int i0, int i1, int i2, int i3);
    GGML_BACKEND_API void    ggml_set_i32_nd(const struct ggml_tensor * tensor, int i0, int i1, int i2, int i3, int32_t value);

    GGML_BACKEND_API float   ggml_get_f32_1d(const struct ggml_tensor * tensor, int i);
    GGML_BACKEND_API void    ggml_set_f32_1d(const struct ggml_tensor * tensor, int i, float value);

    GGML_BACKEND_API float   ggml_get_f32_nd(const struct ggml_tensor * tensor, int i0, int i1, int i2, int i3);
    GGML_BACKEND_API void    ggml_set_f32_nd(const struct ggml_tensor * tensor, int i0, int i1, int i2, int i3, float value);

    GGML_BACKEND_API struct ggml_threadpool *      ggml_threadpool_new           (struct ggml_threadpool_params  * params);
    GGML_BACKEND_API void                          ggml_threadpool_free          (struct ggml_threadpool * threadpool);
    GGML_BACKEND_API int                           ggml_threadpool_get_n_threads (struct ggml_threadpool * threadpool);
    GGML_BACKEND_API void                          ggml_threadpool_pause         (struct ggml_threadpool * threadpool);
    GGML_BACKEND_API void                          ggml_threadpool_resume        (struct ggml_threadpool * threadpool);

    // ggml_graph_plan() has to be called before ggml_graph_compute()
    // when plan.work_size > 0, caller must allocate memory for plan.work_data
    GGML_BACKEND_API struct ggml_cplan ggml_graph_plan(
                  const struct ggml_cgraph * cgraph,
                                       int   n_threads, /* = GGML_DEFAULT_N_THREADS */
                    struct ggml_threadpool * threadpool /* = NULL */ );
    GGML_BACKEND_API enum ggml_status  ggml_graph_compute(struct ggml_cgraph * cgraph, struct ggml_cplan * cplan);

    // same as ggml_graph_compute() but the work data is allocated as a part of the context
    // note: the drawback of this API is that you must have ensured that the context has enough memory for the work data
    GGML_BACKEND_API enum ggml_status  ggml_graph_compute_with_ctx(struct ggml_context * ctx, struct ggml_cgraph * cgraph, int n_threads);

    //
    // system info
    //

    // x86
    GGML_BACKEND_API int ggml_cpu_has_sse3       (void);
    GGML_BACKEND_API int ggml_cpu_has_ssse3      (void);
    GGML_BACKEND_API int ggml_cpu_has_avx        (void);
    GGML_BACKEND_API int ggml_cpu_has_avx_vnni   (void);
    GGML_BACKEND_API int ggml_cpu_has_avx2       (void);
    GGML_BACKEND_API int ggml_cpu_has_bmi2       (void);
    GGML_BACKEND_API int ggml_cpu_has_f16c       (void);
    GGML_BACKEND_API int ggml_cpu_has_fma        (void);
    GGML_BACKEND_API int ggml_cpu_has_avx512     (void);
    GGML_BACKEND_API int ggml_cpu_has_avx512_vbmi(void);
    GGML_BACKEND_API int ggml_cpu_has_avx512_vnni(void);
    GGML_BACKEND_API int ggml_cpu_has_avx512_bf16(void);
    GGML_BACKEND_API int ggml_cpu_has_amx_int8   (void);
    // ARM
    GGML_BACKEND_API int ggml_cpu_has_neon       (void);
    GGML_BACKEND_API int ggml_cpu_has_arm_fma    (void);
    GGML_BACKEND_API int ggml_cpu_has_fp16_va    (void);
    GGML_BACKEND_API int ggml_cpu_has_dotprod    (void);
    GGML_BACKEND_API int ggml_cpu_has_matmul_int8(void);
    GGML_BACKEND_API int ggml_cpu_has_sve        (void);
    GGML_BACKEND_API int ggml_cpu_get_sve_cnt    (void);  // sve vector length in bytes
    GGML_BACKEND_API int ggml_cpu_has_sme        (void);
    // other
    GGML_BACKEND_API int ggml_cpu_has_riscv_v    (void);
    GGML_BACKEND_API int ggml_cpu_get_rvv_vlen   (void);  // risc-v vector length in bytes
    GGML_BACKEND_API int ggml_cpu_has_vsx        (void);
    GGML_BACKEND_API int ggml_cpu_has_vxe        (void);
    GGML_BACKEND_API int ggml_cpu_has_wasm_simd  (void);
    GGML_BACKEND_API int ggml_cpu_has_llamafile  (void);

    // Internal types and functions exposed for tests and benchmarks

    typedef void (*ggml_vec_dot_t)  (int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT x, size_t bx,
                                       const void * GGML_RESTRICT y, size_t by, int nrc);

    struct ggml_type_traits_cpu {
        ggml_from_float_t        from_float;
        ggml_vec_dot_t           vec_dot;
        enum ggml_type           vec_dot_type;
        int64_t                  nrows; // number of rows to process simultaneously
    };

    GGML_BACKEND_API const struct ggml_type_traits_cpu * ggml_get_type_traits_cpu(enum ggml_type type);

    GGML_BACKEND_API void ggml_cpu_init(void);

    //
    // CPU backend
    //

    GGML_BACKEND_API ggml_backend_t ggml_backend_cpu_init(void);

    GGML_BACKEND_API bool ggml_backend_is_cpu                (ggml_backend_t backend);
    GGML_BACKEND_API void ggml_backend_cpu_set_n_threads     (ggml_backend_t backend_cpu, int n_threads);
    GGML_BACKEND_API void ggml_backend_cpu_set_threadpool    (ggml_backend_t backend_cpu, ggml_threadpool_t threadpool);
    GGML_BACKEND_API void ggml_backend_cpu_set_abort_callback(ggml_backend_t backend_cpu, ggml_abort_callback abort_callback, void * abort_callback_data);

    GGML_BACKEND_API void ggml_backend_cpu_set_use_ref(ggml_backend_t backend_cpu, bool use_ref);

    GGML_BACKEND_API ggml_backend_reg_t ggml_backend_cpu_reg(void);

    GGML_BACKEND_API void ggml_cpu_fp32_to_fp32(const float *,       float *, int64_t);
    GGML_BACKEND_API void ggml_cpu_fp32_to_i32 (const float *,     int32_t *, int64_t);
    GGML_BACKEND_API void ggml_cpu_fp32_to_fp16(const float *, ggml_fp16_t *, int64_t);
    GGML_BACKEND_API void ggml_cpu_fp16_to_fp32(const ggml_fp16_t *, float *, int64_t);
    GGML_BACKEND_API void ggml_cpu_fp32_to_bf16(const float *, ggml_bf16_t *, int64_t);
    GGML_BACKEND_API void ggml_cpu_bf16_to_fp32(const ggml_bf16_t *, float *, int64_t);

    //
    // gather_rows_e4m3_e8m0 profiling
    //
    // Counters for GGML_OP_GATHER_ROWS_E4M3_E8M0 only. Model-agnostic: a slot is keyed by the
    // address of the table being gathered, so a caller that gathers from several tables gets
    // one slot per table without this code knowing what a table means.
    //
    // COMPILE-TIME GATE: -DGGML_CPU_PROF, the same switch that gates the INF-70 per-node
    // profiler in ggml-cpu.c and the host-phase profiler in llama-graph.cpp (cmake option
    // GGML_CPU_PROF, OFF by default; ggml/CMakeLists.txt, ggml/src/ggml-cpu/CMakeLists.txt,
    // src/CMakeLists.txt). A non-profiling build is the MEASURED build and must contain no
    // counter code and no instrumentation symbol whatsoever, so the instrument can never be
    // suspected of having moved a number it is absent from. A profiling build is a SEPARATE
    // arm, never the measured one.
    //
    // Inside a profiling build it is still OFF by default at run time: at level 0 the forward
    // pass costs one relaxed load of a static int per thread per node.
    //
    // Level 0: nothing.
    // Level 1: calls, rows, source bytes, thread-0 span, summed per-thread CPU time, and a
    //          log2-spaced histogram of the per-call thread-0 span. Two clock reads per thread
    //          per node.
    // Level 2: level 1 plus minor/major page faults attributed to the op, via
    //          getrusage(RUSAGE_THREAD) around each thread's span. Two extra syscalls per
    //          thread per node -- this PERTURBS what it measures and is a diagnostic mode, not
    //          a steady-state one. Where RUSAGE_THREAD is unavailable it degrades to level 1
    //          and reports fault_source = 0.
    //
#ifdef GGML_CPU_PROF

#define GGML_GATHER_E4M3_PROF_MAX_TABLES 8
#define GGML_GATHER_E4M3_PROF_NBUCKET    48

    struct ggml_gather_e4m3_prof_table {
        const void * table;           // src0->data: the identity of the gathered table
        int64_t n_table_rows;         // src0->ne[1]
        int64_t row_bytes;            // src0->ne[0], the PACKED row width
        int64_t n_calls;              // node executions seen
        int64_t n_rows;               // rows gathered, counting repeats
        int64_t n_bytes_src;          // n_rows * row_bytes: packed bytes dereferenced
        int64_t us_span_ith0;         // summed thread-0 spans (the node-wall proxy)
        int64_t us_cpu;               // summed per-thread spans over every thread
        int64_t n_thread_spans;       // how many thread spans went into us_cpu
        int64_t minflt;               // level 2 only: summed RUSAGE_THREAD minor faults
        int64_t majflt;               // level 2 only: summed RUSAGE_THREAD major faults
        int64_t us_hist[GGML_GATHER_E4M3_PROF_NBUCKET]; // thread-0 span: bucket 0 is 0 us,
                                      // bucket b > 0 covers [2^((b-1)/4), 2^(b/4)) us
    };

    struct ggml_gather_e4m3_prof {
        int     level;
        int     fault_source;         // 0 none, 1 getrusage(RUSAGE_THREAD)
        int     n_tables;             // table slots in use
        int64_t n_calls_unattributed; // calls dropped because every slot was taken
        struct ggml_gather_e4m3_prof_table tables[GGML_GATHER_E4M3_PROF_MAX_TABLES];
    };

    // the level is also taken from the GGML_GATHER_PROF environment variable on first use
    GGML_BACKEND_API void ggml_gather_rows_e4m3_e8m0_prof_set_level(int level);
    GGML_BACKEND_API int  ggml_gather_rows_e4m3_e8m0_prof_get_level(void);

    // snapshot; safe to call while the counters are live, at the cost of a torn read
    GGML_BACKEND_API void ggml_gather_rows_e4m3_e8m0_prof_read (struct ggml_gather_e4m3_prof * out);
    GGML_BACKEND_API void ggml_gather_rows_e4m3_e8m0_prof_reset(void);

#endif // GGML_CPU_PROF

#ifdef __cplusplus
}
#endif
