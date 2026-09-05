#define _CRT_SECURE_NO_DEPRECATE // Disables "unsafe" warnings on Windows
#define _USE_MATH_DEFINES // For M_PI on MSVC

#include "ggml-backend-impl.h"
#include "ggml-backend.h"
#include "traits.h"
#include "ggml-cpu-impl.h"


#include "ggml-impl.h"
#include "quants.h"
#include "ggml-threading.h"
#include "unary-ops.h"
#include "binary-ops.h"
#include "vec.h"
#include "ops.h"
#include "ggml.h"
#include "common.h"

#if defined(_MSC_VER) || defined(__MINGW32__)
#include <malloc.h> // using malloc.h with MSC/MINGW
#elif !defined(__FreeBSD__) && !defined(__NetBSD__) && !defined(__OpenBSD__)
#include <alloca.h>
#endif

#include <assert.h>
#include <errno.h>
#include <time.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <inttypes.h>
#include <stdio.h>
#include <float.h>
#include <limits.h>
#include <stdarg.h>
#include <signal.h>
#if defined(__gnu_linux__)
#include <syscall.h>
#endif

#ifdef GGML_USE_OPENMP
#include <omp.h>
#endif

#if defined(__ARM_FEATURE_SVE) || defined(__ARM_FEATURE_MATMUL_INT8)
#undef GGML_USE_LLAMAFILE
#endif

#ifdef GGML_USE_LLAMAFILE
#include "llamafile/sgemm.h"
#endif

#ifdef GGML_USE_CPU_RISCV64_SPACEMIT
#    include "spacemit/ime.h"
#endif

// Note: once we move threading into a separate C++ file
// will use std::hardware_destructive_interference_size instead of hardcoding it here
// and we'll use C++ attribute syntax.
#define GGML_CACHE_LINE  64

#if defined(__clang__) || defined(__GNUC__)
#define GGML_CACHE_ALIGN __attribute__((aligned(GGML_CACHE_LINE)))
#endif

#if defined(__has_feature)
#if __has_feature(thread_sanitizer)
#define GGML_TSAN_ENABLED 1
#endif
#else  // __has_feature
#if defined(__SANITIZE_THREAD__)
#define GGML_TSAN_ENABLED 1
#endif
#endif // __has_feature

#define UNUSED GGML_UNUSED
#define SWAP(x, y, T) do { T SWAP = x; (x) = y; (y) = SWAP; } while (0)

#ifdef GGML_CPU_PROF
// INF-70: the per-call [mm_prof]/[mmid_prof] lines are extremely noisy (797 + 144 lines per
// decode token) and their fprintf cost dominates the measurement.  They now have their own
// env switch, separate from the aggregate profiler.
static int ggml_cpu_prof_mm_flag = -1;
// INF-70 D0-b: sync events per token, MEASURED rather than derived from the node table.
// Only thread 0 increments, so there is no shared-cacheline traffic on the barrier path.
static uint64_t ggml_cpu_prof_barriers   = 0;   // ggml_barrier() calls seen by thread 0
static int      ggml_cpu_prof_barrier_on = 0;   // set by thread 0 for accumulated graphs only
static inline int ggml_cpu_prof_mm_enabled(void) {
    if (ggml_cpu_prof_mm_flag < 0) {
        ggml_cpu_prof_mm_flag = getenv("GGML_CPU_PROF_MM") != NULL ? 1 : 0;
    }
    return ggml_cpu_prof_mm_flag;
}
#endif

// precomputed f32 table for f16 (256 KB) (simd-mappings.h)
float ggml_table_f32_f16[1 << 16];

// precomputed f32 table for e8m0 half (1 KB) (simd-mappings.h)
float ggml_table_f32_e8m0_half[1 << 8];

// precomputed f32 table for ue4m3 (1 KB) (simd-mappings.h)
float ggml_table_f32_ue4m3[1 << 8];

#if defined(__ARM_ARCH)
struct ggml_arm_arch_features_type {
    int sve_cnt;
} ggml_arm_arch_features = { 0 };
#endif

#if defined(__riscv)
struct ggml_riscv_arch_features_type {
    int rvv_vlen;
} ggml_riscv_arch_features = { 0 };
#endif

#if defined(_WIN32)

#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
    #define NOMINMAX
#endif
#include <windows.h>

#if defined(_MSC_VER) && !defined(__clang__)
#define GGML_CACHE_ALIGN __declspec(align(GGML_CACHE_LINE))

typedef volatile LONG atomic_int;
typedef atomic_int atomic_bool;
typedef atomic_int atomic_flag;

#define ATOMIC_FLAG_INIT 0

typedef enum {
    memory_order_relaxed,
    memory_order_consume,
    memory_order_acquire,
    memory_order_release,
    memory_order_acq_rel,
    memory_order_seq_cst
} memory_order;

static void atomic_store(atomic_int * ptr, LONG val) {
    InterlockedExchange(ptr, val);
}
static void atomic_store_explicit(atomic_int * ptr, LONG val, memory_order mo) {
    // TODO: add support for explicit memory order
    InterlockedExchange(ptr, val);
}
static LONG atomic_load(atomic_int * ptr) {
    return InterlockedCompareExchange(ptr, 0, 0);
}
static LONG atomic_load_explicit(atomic_int * ptr, memory_order mo) {
    // TODO: add support for explicit memory order
    return InterlockedCompareExchange(ptr, 0, 0);
}
static LONG atomic_fetch_add(atomic_int * ptr, LONG inc) {
    return InterlockedExchangeAdd(ptr, inc);
}
static LONG atomic_fetch_add_explicit(atomic_int * ptr, LONG inc, memory_order mo) {
    // TODO: add support for explicit memory order
    return InterlockedExchangeAdd(ptr, inc);
}
static atomic_bool atomic_flag_test_and_set(atomic_flag * ptr) {
    return InterlockedExchange(ptr, 1);
}
static void atomic_flag_clear(atomic_flag * ptr) {
    InterlockedExchange(ptr, 0);
}
static void atomic_thread_fence(memory_order mo) {
    MemoryBarrier();
}
#else // clang
#include <stdatomic.h>
#endif

typedef HANDLE pthread_t;

typedef DWORD thread_ret_t;
static int pthread_create(pthread_t * out, void * unused, thread_ret_t(*func)(void *), void * arg) {
    (void) unused;
    HANDLE handle = CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE) func, arg, 0, NULL);
    if (handle == NULL)
    {
        return EAGAIN;
    }

    *out = handle;
    return 0;
}

static int pthread_join(pthread_t thread, void * unused) {
    (void) unused;
    int ret = (int) WaitForSingleObject(thread, INFINITE);
    CloseHandle(thread);
    return ret;
}

static int sched_yield (void) {
    Sleep (0);
    return 0;
}
#else

#include <pthread.h>
#include <stdatomic.h>
#include <sched.h>
#if defined(__FreeBSD__)
#include <pthread_np.h>
#endif

typedef void * thread_ret_t;

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

#endif

typedef pthread_t ggml_thread_t;

#define GGML_THREADPOOL_N_THREADS_MASK (0xffffU)
#define GGML_THREADPOOL_N_THREADS_BITS (16)

#if defined(__APPLE__)
#include <unistd.h>
#include <mach/mach.h>
#include <TargetConditionals.h>


#endif

static const struct ggml_type_traits_cpu type_traits_cpu[GGML_TYPE_COUNT] = {
    [GGML_TYPE_F32] = {
        .from_float               = (ggml_from_float_t) ggml_cpu_fp32_to_fp32,
        .vec_dot                  = (ggml_vec_dot_t) ggml_vec_dot_f32,
        .vec_dot_type             = GGML_TYPE_F32,
        .nrows                    = 1,
    },
    [GGML_TYPE_F16] = {
        .from_float               = (ggml_from_float_t) ggml_cpu_fp32_to_fp16,
        .vec_dot                  = (ggml_vec_dot_t) ggml_vec_dot_f16,
        .vec_dot_type             = GGML_TYPE_F16,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q1_0] = {
        .from_float               = quantize_row_q1_0,
        .vec_dot                  = ggml_vec_dot_q1_0_q8_0,
        .vec_dot_type             = GGML_TYPE_Q8_0,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q2_0] = {
        .from_float               = quantize_row_q2_0,
        .vec_dot                  = ggml_vec_dot_q2_0_q8_0,
        .vec_dot_type             = GGML_TYPE_Q8_0,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q4_0] = {
        .from_float               = quantize_row_q4_0,
        .vec_dot                  = ggml_vec_dot_q4_0_q8_0,
        .vec_dot_type             = GGML_TYPE_Q8_0,
#if defined (__ARM_FEATURE_MATMUL_INT8)
        .nrows                    = 2,
#else
        .nrows                    = 1,
#endif
    },
    [GGML_TYPE_Q4_1] = {
        .from_float               = quantize_row_q4_1,
        .vec_dot                  = ggml_vec_dot_q4_1_q8_1,
        .vec_dot_type             = GGML_TYPE_Q8_1,
#if defined (__ARM_FEATURE_MATMUL_INT8)
        .nrows                    = 2,
#else
        .nrows                    = 1,
#endif
    },
    [GGML_TYPE_Q5_0] = {
        .from_float               = quantize_row_q5_0,
        .vec_dot                  = ggml_vec_dot_q5_0_q8_0,
        .vec_dot_type             = GGML_TYPE_Q8_0,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q5_1] = {
        .from_float               = quantize_row_q5_1,
        .vec_dot                  = ggml_vec_dot_q5_1_q8_1,
        .vec_dot_type             = GGML_TYPE_Q8_1,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q8_0] = {
        .from_float               = quantize_row_q8_0,
        .vec_dot                  = ggml_vec_dot_q8_0_q8_0,
        .vec_dot_type             = GGML_TYPE_Q8_0,
#if defined (__ARM_FEATURE_MATMUL_INT8)
        .nrows                    = 2,
#else
        .nrows                    = 1,
#endif
    },
    [GGML_TYPE_Q8_1] = {
        .from_float               = quantize_row_q8_1,
        .vec_dot_type             = GGML_TYPE_Q8_1,
        .nrows                    = 1,
    },
    [GGML_TYPE_MXFP4] = {
        .from_float               = quantize_row_mxfp4,
        .vec_dot                  = ggml_vec_dot_mxfp4_q8_0,
        .vec_dot_type             = GGML_TYPE_Q8_0,
        .nrows                    = 1,
    },
    [GGML_TYPE_NVFP4] = {
        .from_float               = quantize_row_nvfp4,
        .vec_dot                  = ggml_vec_dot_nvfp4_q8_0,
        .vec_dot_type             = GGML_TYPE_Q8_0,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q2_K] = {
        .from_float               = quantize_row_q2_K,
        .vec_dot                  = ggml_vec_dot_q2_K_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q3_K] = {
        .from_float               = quantize_row_q3_K,
        .vec_dot                  = ggml_vec_dot_q3_K_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q4_K] = {
        .from_float               = quantize_row_q4_K,
        .vec_dot                  = ggml_vec_dot_q4_K_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
#if defined (__ARM_FEATURE_MATMUL_INT8)
        .nrows                    = 2,
#else
        .nrows                    = 1,
#endif
    },
    [GGML_TYPE_Q5_K] = {
        .from_float               = quantize_row_q5_K,
        .vec_dot                  = ggml_vec_dot_q5_K_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q6_K] = {
        .from_float               = quantize_row_q6_K,
        .vec_dot                  = ggml_vec_dot_q6_K_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
#if defined (__ARM_FEATURE_MATMUL_INT8)
        .nrows                    = 2,
#else
        .nrows                    = 1,
#endif
    },
    [GGML_TYPE_IQ2_XXS] = {
        .from_float               = NULL,
        .vec_dot                  = ggml_vec_dot_iq2_xxs_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ2_XS] = {
        .from_float               = NULL,
        .vec_dot                  = ggml_vec_dot_iq2_xs_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ3_XXS] = {
        // NOTE: from_float for iq3 and iq2_s was removed because these quants require initialization in ggml_quantize_init
        //.from_float               = quantize_row_iq3_xxs,
        .vec_dot                  = ggml_vec_dot_iq3_xxs_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ3_S] = {
        //.from_float               = quantize_row_iq3_s,
        .vec_dot                  = ggml_vec_dot_iq3_s_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ2_S] = {
        //.from_float               = quantize_row_iq2_s,
        .vec_dot                  = ggml_vec_dot_iq2_s_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ1_S] = {
        .from_float               = NULL,
        .vec_dot                  = ggml_vec_dot_iq1_s_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ1_M] = {
        .from_float               = NULL,
        .vec_dot                  = ggml_vec_dot_iq1_m_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ4_NL] = {
        .from_float               = quantize_row_iq4_nl,
        .vec_dot                  = ggml_vec_dot_iq4_nl_q8_0,
        .vec_dot_type             = GGML_TYPE_Q8_0,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ4_XS] = {
        .from_float               = quantize_row_iq4_xs,
        .vec_dot                  = ggml_vec_dot_iq4_xs_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q8_K] = {
        .from_float               = quantize_row_q8_K,
    },
    [GGML_TYPE_BF16] = {
        .from_float               = (ggml_from_float_t) ggml_cpu_fp32_to_bf16,
        .vec_dot                  = (ggml_vec_dot_t) ggml_vec_dot_bf16,
        .vec_dot_type             = GGML_TYPE_BF16,
        .nrows                    = 1,
    },
    [GGML_TYPE_TQ1_0] = {
        .from_float               = quantize_row_tq1_0,
        .vec_dot                  = ggml_vec_dot_tq1_0_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_TQ2_0] = {
        .from_float               = quantize_row_tq2_0,
        .vec_dot                  = ggml_vec_dot_tq2_0_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_I32] = {
        .from_float               = (ggml_from_float_t) ggml_cpu_fp32_to_i32,
    },
};

const struct ggml_type_traits_cpu * ggml_get_type_traits_cpu(enum ggml_type type) {
    return &type_traits_cpu[type];
}

//
// Threading defs
//

typedef pthread_t          ggml_thread_t;

#if defined(_WIN32)

typedef CONDITION_VARIABLE ggml_cond_t;
typedef SRWLOCK            ggml_mutex_t;

#define ggml_mutex_init(m)   InitializeSRWLock(m)
#define ggml_mutex_destroy(m)
#define ggml_mutex_lock(m)   AcquireSRWLockExclusive(m)
#define ggml_mutex_unlock(m) ReleaseSRWLockExclusive(m)
#define ggml_mutex_lock_shared(m)   AcquireSRWLockShared(m)
#define ggml_mutex_unlock_shared(m) ReleaseSRWLockShared(m)

#define ggml_cond_init(c)    InitializeConditionVariable(c)
#define ggml_cond_destroy(c)
#define ggml_cond_wait(c, m) SleepConditionVariableSRW(c, m, INFINITE, CONDITION_VARIABLE_LOCKMODE_SHARED)
#define ggml_cond_broadcast(c) WakeAllConditionVariable(c)

#define ggml_thread_create pthread_create
#define ggml_thread_join   pthread_join

#else

typedef pthread_cond_t     ggml_cond_t;
typedef pthread_mutex_t    ggml_mutex_t;

#define ggml_mutex_init(m)          pthread_mutex_init(m, NULL)
#define ggml_mutex_destroy(m)       pthread_mutex_destroy(m)
#define ggml_mutex_lock(m)          pthread_mutex_lock(m)
#define ggml_mutex_unlock(m)        pthread_mutex_unlock(m)
#define ggml_mutex_lock_shared(m)   pthread_mutex_lock(m)
#define ggml_mutex_unlock_shared(m) pthread_mutex_unlock(m)

#define ggml_lock_init(x)    UNUSED(x)
#define ggml_lock_destroy(x) UNUSED(x)
#if defined(__x86_64__) || (defined(_MSC_VER) && defined(_M_AMD64))
#define ggml_lock_lock(x)    _mm_pause()
#else
#define ggml_lock_lock(x)    UNUSED(x)
#endif
#define ggml_lock_unlock(x)  UNUSED(x)

#define GGML_LOCK_INITIALIZER 0
#define ggml_cond_init(c)      pthread_cond_init(c, NULL)
#define ggml_cond_destroy(c)   pthread_cond_destroy(c)
#define ggml_cond_wait(c, m)   pthread_cond_wait(c, m)
#define ggml_cond_broadcast(c) pthread_cond_broadcast(c)

#define ggml_thread_create pthread_create
#define ggml_thread_join   pthread_join

#endif

// Threadpool def
struct ggml_threadpool {
    ggml_mutex_t mutex;       // mutex for cond.var
    ggml_cond_t  cond;        // cond.var for waiting for new work

    struct ggml_cgraph * cgraph;
    struct ggml_cplan  * cplan;

    // synchronization primitives
    atomic_int n_graph;       // updated when there is work to be done (i.e each graph) holds graph and active thread counts.
    atomic_int GGML_CACHE_ALIGN n_barrier;
    atomic_int GGML_CACHE_ALIGN n_barrier_passed;
    atomic_int GGML_CACHE_ALIGN current_chunk; // currently processing chunk during Mat_Mul, shared between all the threads.

    // these are atomic as an annotation for thread-sanitizer
    atomic_bool stop;         // Used for stopping the threadpool altogether
    atomic_bool pause;        // Used for pausing the threadpool or individual threads
    atomic_int  abort;        // Used for aborting processing of a graph

    struct ggml_compute_state * workers;   // per thread state
    int          n_threads;   // Number of threads in the pool
    int32_t      prio;        // Scheduling priority
    uint32_t     poll;        // Polling level (0 - no polling)

    enum ggml_status ec;
};

// Per-thread state
struct ggml_compute_state {
#ifndef GGML_USE_OPENMP
    ggml_thread_t thrd;
    int  last_graph;
    bool pending;
#endif
    bool cpumask[GGML_MAX_N_THREADS];
    struct ggml_threadpool * threadpool;
    int ith;
};

// Helpers for polling loops
#if defined(__aarch64__) && ( defined(__clang__) || defined(__GNUC__) )
static inline void ggml_thread_cpu_relax(void) {
    __asm__ volatile("yield" ::: "memory");
}
#elif defined(__x86_64__)
static inline void ggml_thread_cpu_relax(void) {
    _mm_pause();
}
#elif defined(__riscv)
static inline void ggml_thread_cpu_relax(void) {
    #ifdef __riscv_zihintpause
        __asm__ __volatile__ ("pause");
    #else
        /* Encoding of the pause instruction */
        __asm__ __volatile__ (".4byte 0x100000F");
    #endif
}
#else
static inline void ggml_thread_cpu_relax(void) {;}
#endif

//
// NUMA support
//

#define GGML_NUMA_MAX_NODES 8
#define GGML_NUMA_MAX_CPUS 512

struct ggml_numa_node {
    uint32_t cpus[GGML_NUMA_MAX_CPUS]; // hardware threads on this node
    uint32_t n_cpus;
};

struct ggml_numa_nodes {
    enum ggml_numa_strategy numa_strategy;
    struct ggml_numa_node nodes[GGML_NUMA_MAX_NODES];
    uint32_t n_nodes;
    uint32_t total_cpus; // hardware threads on system
    uint32_t current_node; // node on which main process is execting
#if defined(__gnu_linux__)
    cpu_set_t cpuset; // cpuset from numactl
#else
    uint32_t cpuset; // no NUMA support outside of Linux at this time. Use a portable datatype
#endif
};

//
// ggml state
//

struct ggml_state {
    struct ggml_numa_nodes numa;
};

static struct ggml_state g_state = {0};

void ggml_barrier(struct ggml_threadpool * tp) {
    int n_threads = atomic_load_explicit(&tp->n_graph, memory_order_relaxed) & GGML_THREADPOOL_N_THREADS_MASK;
#ifdef GGML_CPU_PROF
#ifdef GGML_USE_OPENMP
    if (ggml_cpu_prof_barrier_on && omp_get_thread_num() == 0) {
        ggml_cpu_prof_barriers++;
    }
#endif
#endif
    if (n_threads == 1) {
        return;
    }

#ifdef GGML_USE_OPENMP
    #pragma omp barrier
#else
    int n_passed = atomic_load_explicit(&tp->n_barrier_passed, memory_order_relaxed);

    // enter barrier (full seq-cst fence)
    int n_barrier = atomic_fetch_add_explicit(&tp->n_barrier, 1, memory_order_seq_cst);

    if (n_barrier == (n_threads - 1)) {
        // last thread
        atomic_store_explicit(&tp->n_barrier, 0, memory_order_relaxed);

        // exit barrier (full seq-cst fence)
        atomic_fetch_add_explicit(&tp->n_barrier_passed, 1, memory_order_seq_cst);
        return;
    }

    // wait for other threads
    while (atomic_load_explicit(&tp->n_barrier_passed, memory_order_relaxed) == n_passed) {
        ggml_thread_cpu_relax();
    }

    // exit barrier (full seq-cst fence)
    // TSAN doesn't support standalone fence yet, we use a dummy read-modify-write instead
    #ifdef GGML_TSAN_ENABLED
    atomic_fetch_add_explicit(&tp->n_barrier_passed, 0, memory_order_seq_cst);
    #else
    atomic_thread_fence(memory_order_seq_cst);
    #endif
#endif
}

void ggml_threadpool_chunk_set(struct ggml_threadpool * tp, int value) {
    atomic_store_explicit(&tp->current_chunk, value, memory_order_relaxed);
}

int ggml_threadpool_chunk_add(struct ggml_threadpool * tp, int value) {
    return atomic_fetch_add_explicit(&tp->current_chunk, value, memory_order_relaxed);
}

#if defined(__gnu_linux__)
static cpu_set_t ggml_get_numa_affinity(void) {
    cpu_set_t cpuset;
    pthread_t thread;
    thread = pthread_self();
    CPU_ZERO(&cpuset);
    pthread_getaffinity_np(thread, sizeof(cpu_set_t), &cpuset);
    return cpuset;
}
#else
static uint32_t ggml_get_numa_affinity(void) {
    return 0; // no NUMA support
}
#endif

void ggml_numa_init(enum ggml_numa_strategy numa_flag) {
    if (g_state.numa.n_nodes > 0) {
        fprintf(stderr, "ggml_numa_init: NUMA already initialized\n");

        return;
    }

#if defined(__gnu_linux__)
    struct stat st;
    char path[256];
    int rv;

    // set numa scheme
    g_state.numa.numa_strategy = numa_flag;

    GGML_PRINT_DEBUG("numa strategy %u\n",g_state.numa.numa_strategy);

    g_state.numa.cpuset = ggml_get_numa_affinity();

    // enumerate nodes
    while (g_state.numa.n_nodes < GGML_NUMA_MAX_NODES) {
        rv = snprintf(path, sizeof(path), "/sys/devices/system/node/node%u", g_state.numa.n_nodes);
        GGML_ASSERT(rv > 0 && (unsigned)rv < sizeof(path));
        if (stat(path, &st) != 0) { break; }
        ++g_state.numa.n_nodes;
    }

    // enumerate CPUs
    while (g_state.numa.total_cpus < GGML_NUMA_MAX_CPUS) {
        rv = snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%u", g_state.numa.total_cpus);
        GGML_ASSERT(rv > 0 && (unsigned)rv < sizeof(path));
        if (stat(path, &st) != 0) { break; }
        ++g_state.numa.total_cpus;
    }

    GGML_PRINT_DEBUG("found %u numa nodes, %u CPUs\n", g_state.numa.n_nodes, g_state.numa.total_cpus);

    // figure out which node we're on
    uint current_cpu;
    int getcpu_ret = 0;
#if __GLIBC__ > 2 || (__GLIBC__ == 2 && __GLIBC_MINOR__ > 33) || defined(__COSMOPOLITAN__)
    getcpu_ret = getcpu(&current_cpu, &g_state.numa.current_node);
#else
    // old glibc doesn't have a wrapper for this call. Fall back on direct syscall
#   if !defined(SYS_getcpu) && defined(SYS_get_cpu)
#       define SYS_getcpu SYS_get_cpu // some older glibc versions use this name
#   endif
    getcpu_ret = syscall(SYS_getcpu, &current_cpu, &g_state.numa.current_node);
#endif

    if (g_state.numa.n_nodes < 1 || g_state.numa.total_cpus < 1 || getcpu_ret != 0) {
        g_state.numa.n_nodes = 0;
        return;
    }

    GGML_PRINT_DEBUG("found our process on numa node %u, CPU %u\n", g_state.numa.current_node, current_cpu);

    for (uint32_t n = 0; n < g_state.numa.n_nodes; ++n) {
        struct ggml_numa_node * node = &g_state.numa.nodes[n];
        GGML_PRINT_DEBUG("CPUs on node %u:", n);
        node->n_cpus = 0;
        for (uint32_t c = 0; c < g_state.numa.total_cpus; ++c) {
            rv = snprintf(path, sizeof(path), "/sys/devices/system/node/node%u/cpu%u", n, c);
            GGML_ASSERT(rv > 0 && (unsigned)rv < sizeof(path));
            if (stat(path, &st) == 0) {
                node->cpus[node->n_cpus++] = c;
                GGML_PRINT_DEBUG(" %u", c);
            }
        }
        GGML_PRINT_DEBUG("\n");
    }

    if (ggml_is_numa()) {
        FILE *fptr = fopen("/proc/sys/kernel/numa_balancing", "r");
        if (fptr != NULL) {
            char buf[42];
            if (fgets(buf, sizeof(buf), fptr) && strncmp(buf, "0\n", sizeof(buf)) != 0) {
                GGML_LOG_WARN("/proc/sys/kernel/numa_balancing is enabled, this has been observed to impair performance\n");
            }
            fclose(fptr);
        }
    }
#else
    UNUSED(numa_flag);
    // TODO
#endif
}

bool ggml_is_numa(void) {
    return g_state.numa.n_nodes > 1;
}

#if defined(__ARM_ARCH)
#if defined(__aarch64__) && defined(__ARM_FEATURE_SVE)
#include <arm_sve.h>
static void ggml_init_arm_arch_features(void) {
    ggml_arm_arch_features.sve_cnt = svcntb();
}
#else
static void ggml_init_arm_arch_features(void) {}
#endif
#endif // __ARM_ARCH

#if defined(__riscv) && defined(__riscv_v_intrinsic)
#include <riscv_vector.h>
static void ggml_init_riscv_arch_features(void) {
    ggml_riscv_arch_features.rvv_vlen = __riscv_vlenb();
}
#else
static void ggml_init_riscv_arch_features(void) {}
#endif

struct ggml_tensor * ggml_new_i32(struct ggml_context * ctx, int32_t value) {
    GGML_ASSERT(!ggml_get_no_alloc(ctx));

    struct ggml_tensor * result = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);

    ggml_set_i32(result, value);

    return result;
}

struct ggml_tensor * ggml_new_f32(struct ggml_context * ctx, float value) {
    GGML_ASSERT(!ggml_get_no_alloc(ctx));

    struct ggml_tensor * result = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);

    ggml_set_f32(result, value);

    return result;
}

struct ggml_tensor * ggml_set_i32 (struct ggml_tensor * tensor, int32_t value) {
    const int n     = ggml_nrows(tensor);
    const int nc    = tensor->ne[0];
    const size_t n1 = tensor->nb[1];

    char * const data = tensor->data;

    switch (tensor->type) {
        case GGML_TYPE_I8:
            {
                assert(tensor->nb[0] == sizeof(int8_t));
                for (int i = 0; i < n; i++) {
                    ggml_vec_set_i8(nc, (int8_t *)(data + i*n1), value);
                }
            } break;
        case GGML_TYPE_I16:
            {
                assert(tensor->nb[0] == sizeof(int16_t));
                for (int i = 0; i < n; i++) {
                    ggml_vec_set_i16(nc, (int16_t *)(data + i*n1), value);
                }
            } break;
        case GGML_TYPE_I32:
            {
                assert(tensor->nb[0] == sizeof(int32_t));
                for (int i = 0; i < n; i++) {
                    ggml_vec_set_i32(nc, (int32_t *)(data + i*n1), value);
                }
            } break;
        case GGML_TYPE_F16:
            {
                assert(tensor->nb[0] == sizeof(ggml_fp16_t));
                for (int i = 0; i < n; i++) {
                    ggml_vec_set_f16(nc, (ggml_fp16_t *)(data + i*n1), GGML_CPU_FP32_TO_FP16(value));
                }
            } break;
        case GGML_TYPE_BF16:
            {
                assert(tensor->nb[0] == sizeof(ggml_fp16_t));
                for (int i = 0; i < n; i++) {
                    ggml_vec_set_bf16(nc, (ggml_bf16_t *)(data + i*n1), GGML_FP32_TO_BF16(value));
                }
            } break;
        case GGML_TYPE_F32:
            {
                assert(tensor->nb[0] == sizeof(float));
                for (int i = 0; i < n; i++) {
                    ggml_vec_set_f32(nc, (float *)(data + i*n1), value);
                }
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }

    return tensor;
}

struct ggml_tensor * ggml_set_f32(struct ggml_tensor * tensor, float value) {
    const int n     = ggml_nrows(tensor);
    const int nc    = tensor->ne[0];
    const size_t n1 = tensor->nb[1];

    char * const data = tensor->data;

    switch (tensor->type) {
        case GGML_TYPE_I8:
            {
                assert(tensor->nb[0] == sizeof(int8_t));
                for (int i = 0; i < n; i++) {
                    ggml_vec_set_i8(nc, (int8_t *)(data + i*n1), value);
                }
            } break;
        case GGML_TYPE_I16:
            {
                assert(tensor->nb[0] == sizeof(int16_t));
                for (int i = 0; i < n; i++) {
                    ggml_vec_set_i16(nc, (int16_t *)(data + i*n1), value);
                }
            } break;
        case GGML_TYPE_I32:
            {
                assert(tensor->nb[0] == sizeof(int32_t));
                for (int i = 0; i < n; i++) {
                    ggml_vec_set_i32(nc, (int32_t *)(data + i*n1), value);
                }
            } break;
        case GGML_TYPE_F16:
            {
                assert(tensor->nb[0] == sizeof(ggml_fp16_t));
                for (int i = 0; i < n; i++) {
                    ggml_vec_set_f16(nc, (ggml_fp16_t *)(data + i*n1), GGML_CPU_FP32_TO_FP16(value));
                }
            } break;
        case GGML_TYPE_BF16:
            {
                assert(tensor->nb[0] == sizeof(ggml_bf16_t));
                for (int i = 0; i < n; i++) {
                    ggml_vec_set_bf16(nc, (ggml_bf16_t *)(data + i*n1), GGML_FP32_TO_BF16(value));
                }
            } break;
        case GGML_TYPE_F32:
            {
                assert(tensor->nb[0] == sizeof(float));
                for (int i = 0; i < n; i++) {
                    ggml_vec_set_f32(nc, (float *)(data + i*n1), value);
                }
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }

    return tensor;
}

int32_t ggml_get_i32_1d(const struct ggml_tensor * tensor, int i) {
    if (!ggml_is_contiguous(tensor)) {
        int64_t id[4] = { 0, 0, 0, 0 };
        ggml_unravel_index(tensor, i, &id[0], &id[1], &id[2], &id[3]);
        return ggml_get_i32_nd(tensor, id[0], id[1], id[2], id[3]);
    }
    switch (tensor->type) {
        case GGML_TYPE_I8:
            {
                GGML_ASSERT(tensor->nb[0] == sizeof(int8_t));
                return ((int8_t *)(tensor->data))[i];
            }
        case GGML_TYPE_I16:
            {
                GGML_ASSERT(tensor->nb[0] == sizeof(int16_t));
                return ((int16_t *)(tensor->data))[i];
            }
        case GGML_TYPE_I32:
            {
                GGML_ASSERT(tensor->nb[0] == sizeof(int32_t));
                return ((int32_t *)(tensor->data))[i];
            }
        case GGML_TYPE_F16:
            {
                GGML_ASSERT(tensor->nb[0] == sizeof(ggml_fp16_t));
                return GGML_CPU_FP16_TO_FP32(((ggml_fp16_t *)(tensor->data))[i]);
            }
        case GGML_TYPE_BF16:
            {
                GGML_ASSERT(tensor->nb[0] == sizeof(ggml_bf16_t));
                return GGML_BF16_TO_FP32(((ggml_bf16_t *)(tensor->data))[i]);
            }
        case GGML_TYPE_F32:
            {
                GGML_ASSERT(tensor->nb[0] == sizeof(float));
                return ((float *)(tensor->data))[i];
            }
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}

void ggml_set_i32_1d(const struct ggml_tensor * tensor, int i, int32_t value) {
    if (!ggml_is_contiguous(tensor)) {
        int64_t id[4] = { 0, 0, 0, 0 };
        ggml_unravel_index(tensor, i, &id[0], &id[1], &id[2], &id[3]);
        ggml_set_i32_nd(tensor, id[0], id[1], id[2], id[3], value);
        return;
    }
    switch (tensor->type) {
        case GGML_TYPE_I8:
            {
                GGML_ASSERT(tensor->nb[0] == sizeof(int8_t));
                ((int8_t *)(tensor->data))[i] = value;
            } break;
        case GGML_TYPE_I16:
            {
                GGML_ASSERT(tensor->nb[0] == sizeof(int16_t));
                ((int16_t *)(tensor->data))[i] = value;
            } break;
        case GGML_TYPE_I32:
            {
                GGML_ASSERT(tensor->nb[0] == sizeof(int32_t));
                ((int32_t *)(tensor->data))[i] = value;
            } break;
        case GGML_TYPE_F16:
            {
                GGML_ASSERT(tensor->nb[0] == sizeof(ggml_fp16_t));
                ((ggml_fp16_t *)(tensor->data))[i] = GGML_CPU_FP32_TO_FP16(value);
            } break;
        case GGML_TYPE_BF16:
            {
                GGML_ASSERT(tensor->nb[0] == sizeof(ggml_bf16_t));
                ((ggml_bf16_t *)(tensor->data))[i] = GGML_FP32_TO_BF16(value);
            } break;
        case GGML_TYPE_F32:
            {
                GGML_ASSERT(tensor->nb[0] == sizeof(float));
                ((float *)(tensor->data))[i] = value;
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}

int32_t ggml_get_i32_nd(const struct ggml_tensor * tensor, int i0, int i1, int i2, int i3) {
    void * data   = (char *) tensor->data + i0*tensor->nb[0] + i1*tensor->nb[1] + i2*tensor->nb[2] + i3*tensor->nb[3];
    switch (tensor->type) {
        case GGML_TYPE_I8:
            return ((int8_t *) data)[0];
        case GGML_TYPE_I16:
            return ((int16_t *) data)[0];
        case GGML_TYPE_I32:
            return ((int32_t *) data)[0];
        case GGML_TYPE_F16:
            return GGML_CPU_FP16_TO_FP32(((ggml_fp16_t *) data)[0]);
        case GGML_TYPE_BF16:
            return GGML_BF16_TO_FP32(((ggml_bf16_t *) data)[0]);
        case GGML_TYPE_F32:
            return ((float *) data)[0];
        default:
            GGML_ABORT("fatal error");
    }
}

void ggml_set_i32_nd(const struct ggml_tensor * tensor, int i0, int i1, int i2, int i3, int32_t value) {
    void * data   = (char *) tensor->data + i0*tensor->nb[0] + i1*tensor->nb[1] + i2*tensor->nb[2] + i3*tensor->nb[3];
    switch (tensor->type) {
        case GGML_TYPE_I8:
            {
                ((int8_t *)(data))[0] = value;
            } break;
        case GGML_TYPE_I16:
            {
                ((int16_t *)(data))[0] = value;
            } break;
        case GGML_TYPE_I32:
            {
                ((int32_t *)(data))[0] = value;
            } break;
        case GGML_TYPE_F16:
            {
                ((ggml_fp16_t *)(data))[0] = GGML_CPU_FP32_TO_FP16(value);
            } break;
        case GGML_TYPE_BF16:
            {
                ((ggml_bf16_t *)(data))[0] = GGML_FP32_TO_BF16(value);
            } break;
        case GGML_TYPE_F32:
            {
                ((float *)(data))[0] = value;
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}

float ggml_get_f32_1d(const struct ggml_tensor * tensor, int i) {
    if (!ggml_is_contiguous(tensor)) {
        int64_t id[4] = { 0, 0, 0, 0 };
        ggml_unravel_index(tensor, i, &id[0], &id[1], &id[2], &id[3]);
        return ggml_get_f32_nd(tensor, id[0], id[1], id[2], id[3]);
    }
    switch (tensor->type) {
        case GGML_TYPE_I8:
            {
                return ((int8_t *)(tensor->data))[i];
            }
        case GGML_TYPE_I16:
            {
                return ((int16_t *)(tensor->data))[i];
            }
        case GGML_TYPE_I32:
            {
                return ((int32_t *)(tensor->data))[i];
            }
        case GGML_TYPE_F16:
            {
                return GGML_CPU_FP16_TO_FP32(((ggml_fp16_t *)(tensor->data))[i]);
            }
        case GGML_TYPE_BF16:
            {
                return GGML_BF16_TO_FP32(((ggml_bf16_t *)(tensor->data))[i]);
            }
        case GGML_TYPE_F32:
            {
                return ((float *)(tensor->data))[i];
            }
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}

void ggml_set_f32_1d(const struct ggml_tensor * tensor, int i, float value) {
    if (!ggml_is_contiguous(tensor)) {
        int64_t id[4] = { 0, 0, 0, 0 };
        ggml_unravel_index(tensor, i, &id[0], &id[1], &id[2], &id[3]);
        ggml_set_f32_nd(tensor, id[0], id[1], id[2], id[3], value);
        return;
    }
    switch (tensor->type) {
        case GGML_TYPE_I8:
            {
                ((int8_t *)(tensor->data))[i] = value;
            } break;
        case GGML_TYPE_I16:
            {
                ((int16_t *)(tensor->data))[i] = value;
            } break;
        case GGML_TYPE_I32:
            {
                ((int32_t *)(tensor->data))[i] = value;
            } break;
        case GGML_TYPE_F16:
            {
                ((ggml_fp16_t *)(tensor->data))[i] = GGML_CPU_FP32_TO_FP16(value);
            } break;
        case GGML_TYPE_BF16:
            {
                ((ggml_bf16_t *)(tensor->data))[i] = GGML_FP32_TO_BF16(value);
            } break;
        case GGML_TYPE_F32:
            {
                ((float *)(tensor->data))[i] = value;
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}

float ggml_get_f32_nd(const struct ggml_tensor * tensor, int i0, int i1, int i2, int i3) {
    void * data   = (char *) tensor->data + i0*tensor->nb[0] + i1*tensor->nb[1] + i2*tensor->nb[2] + i3*tensor->nb[3];
    switch (tensor->type) {
        case GGML_TYPE_I8:
            return ((int8_t *) data)[0];
        case GGML_TYPE_I16:
            return ((int16_t *) data)[0];
        case GGML_TYPE_I32:
            return ((int32_t *) data)[0];
        case GGML_TYPE_F16:
            return GGML_CPU_FP16_TO_FP32(((ggml_fp16_t *) data)[0]);
        case GGML_TYPE_BF16:
            return GGML_BF16_TO_FP32(((ggml_bf16_t *) data)[0]);
        case GGML_TYPE_F32:
            return ((float *) data)[0];
        default:
            GGML_ABORT("fatal error");
    }
}

void ggml_set_f32_nd(const struct ggml_tensor * tensor, int i0, int i1, int i2, int i3, float value) {
    void * data   = (char *) tensor->data + i0*tensor->nb[0] + i1*tensor->nb[1] + i2*tensor->nb[2] + i3*tensor->nb[3];
    switch (tensor->type) {
        case GGML_TYPE_I8:
            {
                ((int8_t *)(data))[0] = value;
            } break;
        case GGML_TYPE_I16:
            {
                ((int16_t *)(data))[0] = value;
            } break;
        case GGML_TYPE_I32:
            {
                ((int32_t *)(data))[0] = value;
            } break;
        case GGML_TYPE_F16:
            {
                ((ggml_fp16_t *)(data))[0] = GGML_CPU_FP32_TO_FP16(value);
            } break;
        case GGML_TYPE_BF16:
            {
                ((ggml_bf16_t *)(data))[0] = GGML_FP32_TO_BF16(value);
            } break;
        case GGML_TYPE_F32:
            {
                ((float *)(data))[0] = value;
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}

////////////////////////////////////////////////////////////////////////////////

// ggml_compute_forward_mul_mat

// INF-70 D1: the mul_mat chunk plan, factored out unchanged so the batch-1 fast path
// can decide *before* quantizing src1 whether the shared chunk counter is ever read.
static inline void ggml_mul_mat_chunk_plan(
        const int64_t nr0, const int64_t nr1, const int nth,
        int64_t * nchunk0, int64_t * nchunk1) {

    // Now select a reasonable chunk size.
    int chunk_size = 16;

    // We need to step up the size if it's small
    if (nr0 == 1 || nr1 == 1) {
        chunk_size = 64;
    }

    // distribute the work across the inner or outer loop based on which one is larger
    // The number of chunks in the 0/1 dim.
    // CEIL(nr0/chunk_size)
    int64_t c0 = (nr0 + chunk_size - 1) / chunk_size;
    int64_t c1 = (nr1 + chunk_size - 1) / chunk_size;

    // If the chunking is poor for the number of threads on this setup, scrap the whole plan.  Re-chunk it by thread.
    //   Also, chunking by thread was measured to have perform better on NUMA systems.  See https://github.com/ggml-org/llama.cpp/pull/6915
    //   In theory, chunking should be just as useful on NUMA and non NUMA systems, but testing disagreed with that.
    if (c0 * c1 < nth * 4 || ggml_is_numa()) {
        // distribute the thread work across the inner or outer loop based on which one is larger
        c0 = nr0 > nr1 ? nth : 1; // parallelize by src0 rows
        c1 = nr0 > nr1 ? 1 : nth; // parallelize by src1 rows
    }

    *nchunk0 = c0;
    *nchunk1 = c1;
}

static void ggml_compute_forward_mul_mat_one_chunk(
    const struct ggml_compute_params * params,
    struct ggml_tensor * dst,
    const enum ggml_type type,
    const int64_t num_rows_per_vec_dot,
    const int64_t ir0_start,
    const int64_t ir0_end,
    const int64_t ir1_start,
    const int64_t ir1_end,
    const void * src1_wdata) {

    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_BINARY_OP_LOCALS

    const bool src1_cont = ggml_is_contiguous(src1);

    ggml_vec_dot_t const vec_dot      = type_traits_cpu[type].vec_dot;
    enum ggml_type const vec_dot_type = type_traits_cpu[type].vec_dot_type;

    // broadcast factors
    const int64_t r2 = ne12 / ne02;
    const int64_t r3 = ne13 / ne03;

    //printf("ir0_start = %6lld, ir0_end = %6lld, ir1_start = %6lld, ir1_end = %6lld\n", ir0_start, ir0_end, ir1_start, ir1_end);

    // threads with no work simply yield (not sure if it helps)
    if (ir0_start >= ir0_end || ir1_start >= ir1_end) {
        return;
    }

    // INF-70 D1: src1_wdata is params->wdata on the shared path and this thread's private
    // slice on the batch-1 path; identical bytes either way.
    const void * wdata = (src1->type == vec_dot_type) ? src1->data : src1_wdata;
    const size_t row_size = ggml_row_size(vec_dot_type, ne10);

    assert(ne12 % ne02 == 0);
    assert(ne13 % ne03 == 0);

    // block-tiling attempt
    const int64_t blck_0 = 16;
    const int64_t blck_1 = 16;

    const size_t src1_col_stride = src1_cont || src1->type != vec_dot_type ? row_size : nb11;

    // attempt to reduce false-sharing (does not seem to make a difference)
    // 16 * 2, accounting for mmla kernels
    float tmp[32];

    for (int64_t iir1 = ir1_start; iir1 < ir1_end; iir1 += blck_1) {
        for (int64_t iir0 = ir0_start; iir0 < ir0_end; iir0 += blck_0) {
            for (int64_t ir1 = iir1; ir1 < iir1 + blck_1 && ir1 < ir1_end; ir1 += num_rows_per_vec_dot) {
                const int64_t i13 = (ir1 / (ne12 * ne1));
                const int64_t i12 = (ir1 - i13 * ne12 * ne1) / ne1;
                const int64_t i11 = (ir1 - i13 * ne12 * ne1 - i12 * ne1);

                // broadcast src0 into src1
                const int64_t i03 = i13 / r3;
                const int64_t i02 = i12 / r2;

                const int64_t i1 = i11;
                const int64_t i2 = i12;
                const int64_t i3 = i13;

                const char * src0_row = (const char*)src0->data + (0 + i02 * nb02 + i03 * nb03);

                // desc: when src1 is not a contiguous memory block we have to calculate the offset using the strides
                //       if it is, then we have either copied the data to params->wdata and made it contiguous or we are using
                //       the original src1 data pointer, so we should index using the indices directly
                // TODO: this is a bit of a hack, we should probably have a better way to handle this
                const char * src1_col = (const char*)wdata +
                    (src1_cont || src1->type != vec_dot_type
                        ? (i11 + i12 * ne11 + i13 * ne12 * ne11) * row_size
                        : (i11 * nb11 + i12 * nb12 + i13 * nb13));
                float * dst_col = (float*)((char*)dst->data + (i1 * nb1 + i2 * nb2 + i3 * nb3));

                //for (int64_t ir0 = iir0; ir0 < iir0 + blck_0 && ir0 < ir0_end; ++ir0) {
                //    vec_dot(ne00, &dst_col[ir0], src0_row + ir0*nb01, src1_col);
                //}

                for (int64_t ir0 = iir0; ir0 < iir0 + blck_0 && ir0 < ir0_end; ir0 += num_rows_per_vec_dot) {
                    vec_dot(ne00, &tmp[ir0 - iir0], (num_rows_per_vec_dot > 1 ? 16 : 0), src0_row + ir0 * nb01, (num_rows_per_vec_dot > 1 ? nb01 : 0), src1_col, (num_rows_per_vec_dot > 1 ? src1_col_stride : 0), num_rows_per_vec_dot);
                }

                for (int cn = 0; cn < num_rows_per_vec_dot; ++cn) {
                    memcpy(&dst_col[iir0 + cn * nb1 / nb0], tmp + (cn * 16), (MIN(iir0 + blck_0, ir0_end) - iir0) * sizeof(float));
                }
            }
        }
    }
}

// INF-70 BE-1: true when this mul_mat is a small batch that must be computed row-exactly,
// i.e. every output row bit-equal to the corresponding single-token (ne11 == 1) decode.
static inline bool ggml_cpu_rowexact_batch(int64_t ne11) {
    return ne11 > 1 && ne11 <= ggml_cpu_rowexact_n();
}

void ggml_compute_forward_mul_mat(
        const struct ggml_compute_params * params,
              struct ggml_tensor * dst) {

    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];

    const int32_t hint = ggml_get_op_params_i32(dst, 1);
    if (hint == GGML_HINT_SRC0_IS_HADAMARD && !params->use_ref) {
        ggml_compute_forward_fwht(params, dst);
        return;
    }

    GGML_TENSOR_BINARY_OP_LOCALS

    const int ith = params->ith;
    const int nth = params->nth;

    enum ggml_type           const vec_dot_type         = type_traits_cpu[src0->type].vec_dot_type;
    ggml_from_float_t        const from_float           = type_traits_cpu[vec_dot_type].from_float;
    int64_t                  const vec_dot_num_rows     = type_traits_cpu[src0->type].nrows;

    GGML_ASSERT(ne0 == ne01);
    GGML_ASSERT(ne1 == ne11);
    GGML_ASSERT(ne2 == ne12);
    GGML_ASSERT(ne3 == ne13);

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == ggml_type_size(src0->type));
    GGML_ASSERT(nb10 == ggml_type_size(src1->type));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == sizeof(float));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    // nb01 >= nb00 - src0 is not transposed
    //   compute by src0 rows

#ifdef GGML_CPU_PROF
    const int64_t mm_t0 = ggml_cpu_prof_mm_enabled() ? ggml_time_us() : 0;
#endif

#if defined(GGML_USE_IQK_MULMAT)
    // iqk port: fast quantized GEMM (ik_llama kernels) for supported quant types.
    // Runtime-gated by env GGML_IQK=1; returns false (falls through) otherwise or
    // for unsupported types/dims. Handles its own src1 quantization + barrier.
    if (ggml_iqk_try_mul_mat(params, dst)) {
        return;
    }
#endif

    // TODO: extract to "extra_op"
#if GGML_USE_LLAMAFILE
    // broadcast factors
    const int64_t r2 = ne12 / ne02;
    const int64_t r3 = ne13 / ne03;

    const bool src1_cont = ggml_is_contiguous(src1);

    // INF-70 GDN-ROWEXACT / BE-1: for 1 < ne11 <= GGML_ROWEXACT_N every output row must be
    // bit-equal to the corresponding ne11 == 1 single-token decode (the tiled kernels pick a
    // different accumulation order per tile shape, a 1-ulp difference that MoE top-k amplifies
    // into different expert selections).
    //
    // The original attempt ran the tinyBLAS GEMM one src1 column at a time. That could never
    // work: llamafile_sgemm hard-refuses n < 2 ("only enable sgemm for prompt processing",
    // llamafile/sgemm.cpp), so the loop failed on its FIRST column and fell straight through
    // into the full-batch tinyBLAS GEMM below -- GGML_ROWEXACT_N was a silent no-op for every
    // tinyBLAS mul_mat, including the F32 MoE router ffn_gate_inp (INF-70 batch-envelope).
    //
    // The correct answer is to skip the tinyBLAS section entirely: the single-token decode is
    // refused by llamafile_sgemm for exactly these shapes too and runs the generic vec_dot
    // mul_mat below, so the generic path IS the row-exact reference, not an approximation of it.
    if (src1_cont && ggml_cpu_rowexact_batch(ne11)) {
        goto UseGgmlGemm1;
    }

    if (src1_cont) {
        for (int64_t i13 = 0; i13 < ne13; i13++)
            for (int64_t i12 = 0; i12 < ne12; i12++)
                if (!llamafile_sgemm(params,
                                     ne01, ne11, ne00/ggml_blck_size(src0->type),
                                     (const char *)src0->data + i12/r2*nb02 + i13/r3*nb03,
                                     nb01/ggml_type_size(src0->type),
                                     (const char *)src1->data + i12*nb12 + i13*nb13,
                                     nb11/ggml_type_size(src1->type),
                                     (char *)dst->data + i12*nb2 + i13*nb3,
                                     nb1/ggml_type_size(dst->type),
                                     src0->type,
                                     src1->type,
                                     dst->type))
                    goto UseGgmlGemm1;
        return;
    }
UseGgmlGemm1:;
#endif

    // INF-70 D1: batch-1 fast path — drop the internal barrier.
    //
    // With a single src1 row the stock code splits that row's blocks across all nth
    // threads (10 Q8_K blocks for 48 threads: 38 threads quantize nothing) and then pays
    // a full-team ggml_barrier. Instead every thread converts the WHOLE row into its own
    // private slice of wdata and reads its own copy in the dot loop, so nothing has to be
    // published between threads and the barrier disappears.
    //
    // The barrier also publishes threadpool->current_chunk, initialised by ith==0. It is
    // only ever *read* when nth < nchunk0*nchunk1 (otherwise the chunk loop breaks after
    // the first chunk), so the fast path additionally requires nth >= nchunk0*nchunk1 —
    // then the counter is provably dead and needs no initialisation. Bit-exactness: same
    // from_float over the same whole row, same chunk plan, same vec_dot call order.
    const int64_t src1_nrows_total = ne11*ne12*ne13;

    int64_t nchunk0 = 0, nchunk1 = 0;
    ggml_mul_mat_chunk_plan(ne0, ne1*ne2*ne3, nth, &nchunk0, &nchunk1);

    const bool mm_batch1 =
        src1_nrows_total == 1 &&
        nth >= nchunk0*nchunk1 &&
        (src1->type == vec_dot_type ||
         params->wsize >= (size_t) nth * ggml_row_size(vec_dot_type, ne10));

    const void * src1_wdata = params->wdata;

    if (src1->type != vec_dot_type) {
        char * wdata = params->wdata;

        const size_t nbw0 = ggml_type_size(vec_dot_type);
        const size_t nbw1 = ggml_row_size(vec_dot_type, ne10);
        const size_t nbw2 = nbw1*ne11;
        const size_t nbw3 = nbw2*ne12;

        assert(params->wsize >= ne13*nbw3);
        GGML_ASSERT(src1->type == GGML_TYPE_F32);

        if (mm_batch1) {
            // ne11 == ne12 == ne13 == 1, so the whole of src1 is one row at offset 0.
            char * wdata_priv = wdata + (size_t) ith * nbw1;
            from_float((const float *) src1->data, (void *) wdata_priv, ne10);
            src1_wdata = wdata_priv;
        } else {

    #if 0
        for (int64_t i13 = 0; i13 < ne13; ++i13) {
            for (int64_t i12 = 0; i12 < ne12; ++i12) {
                for (int64_t i11 = ith; i11 < ne11; i11 += nth) {
                    from_float((float *)((char *) src1->data + i13*nb13 + i12*nb12 + i11*nb11),
                               (void *)               (wdata + i13*nbw3 + i12*nbw2 + i11*nbw1),
                                ne10);
                }
            }
        }
    #else
        for (int64_t i13 = 0; i13 < ne13; ++i13) {
            for (int64_t i12 = 0; i12 < ne12; ++i12) {
                for (int64_t i11 = 0; i11 < ne11; ++i11) {
                    size_t bs = ggml_blck_size(vec_dot_type);
                    int64_t ne10_block_start = (ith * ne10/bs) / nth;
                    int64_t ne10_block_end   = ((ith + 1) * ne10/bs) / nth;
                    from_float((float *)((char *) src1->data + i13*nb13 + i12*nb12 + i11*nb11 + ne10_block_start*bs*nb10),
                               (void *)               (wdata + i13*nbw3 + i12*nbw2 + i11*nbw1 + ne10_block_start*nbw0),
                               (ne10_block_end - ne10_block_start) * bs);
                }
            }
        }
    #endif
        } // !mm_batch1
    }

    if (!mm_batch1) {
        if (ith == 0) {
            // Every thread starts at ith, so the first unprocessed chunk is nth.  This save a bit of coordination right at the start.
            atomic_store_explicit(&params->threadpool->current_chunk, nth, memory_order_relaxed);
        }

        ggml_barrier(params->threadpool);
    }

#if GGML_USE_LLAMAFILE
    // INF-70 BE-1: same row-exactness guard as the first tinyBLAS block above.
    if (src1->type != vec_dot_type && !ggml_cpu_rowexact_batch(ne11)) {
        const void* wdata = (src1->type == vec_dot_type) ? src1->data : src1_wdata;
        const size_t row_size = ggml_row_size(vec_dot_type, ne10);

        for (int64_t i13 = 0; i13 < ne13; i13++)
            for (int64_t i12 = 0; i12 < ne12; i12++)
                if (!llamafile_sgemm(params,
                                     ne01, ne11, ne00/ggml_blck_size(src0->type),
                                     (const char *)src0->data + i12/r2*nb02 + i13/r3*nb03,
                                     nb01/ggml_type_size(src0->type),
                                     (const char *)wdata + (i12*ne11 + i13*ne12*ne11)*row_size,
                                     row_size/ggml_type_size(vec_dot_type),
                                     (char *)dst->data + i12*nb2 + i13*nb3,
                                     nb1/ggml_type_size(dst->type),
                                     src0->type,
                                     vec_dot_type,
                                     dst->type))
                    goto UseGgmlGemm2;
        return;
    }
UseGgmlGemm2:;
#endif

    // This is the size of the first dimension of the result, so we can iterate that way. (see the ASSERT above, these are the same numbers)
    const int64_t nr0 = ne0;

    // This is the size of the rest of the dimensions of the result
    const int64_t nr1 = ne1 * ne2 * ne3;

    // INF-70 D1: nchunk0/nchunk1 were computed above by ggml_mul_mat_chunk_plan(ne0, ne1*ne2*ne3, nth)
    // — the identical formula, hoisted so the batch-1 decision can be made before quantizing.
    GGML_ASSERT(nchunk0 > 0 && nchunk1 > 0);

    // The number of elements in each chunk
    const int64_t dr0 = (nr0 + nchunk0 - 1) / nchunk0;
    const int64_t dr1 = (nr1 + nchunk1 - 1) / nchunk1;

    // The first chunk comes from our thread_id, the rest will get auto-assigned.
    int current_chunk = ith;

    while (current_chunk < nchunk0 * nchunk1) {
        const int64_t ith0 = current_chunk % nchunk0;
        const int64_t ith1 = current_chunk / nchunk0;

        const int64_t ir0_start = dr0 * ith0;
        const int64_t ir0_end = MIN(ir0_start + dr0, nr0);

        const int64_t ir1_start = dr1 * ith1;
        const int64_t ir1_end = MIN(ir1_start + dr1, nr1);

        // dot kernels can handle 1 row and col at a time, but mmla kernels can process 2 rows and cols
        int64_t num_rows_per_vec_dot = vec_dot_num_rows;

        // these checks are needed to avoid crossing dim1 boundaries
        // can be optimized, but the logic would become more complicated, so keeping it like this for simplicity
        if ((nr0 % 2 != 0) || (ne11 % 2 != 0) || ((ir0_end - ir0_start) % 2 != 0) || ((ir1_end - ir1_start) % 2 != 0)) {
            num_rows_per_vec_dot = 1;
        }
        ggml_compute_forward_mul_mat_one_chunk(params, dst, src0->type, num_rows_per_vec_dot, ir0_start, ir0_end, ir1_start, ir1_end, src1_wdata);

        if (nth >= nchunk0 * nchunk1) {
            break;
        }

        current_chunk = atomic_fetch_add_explicit(&params->threadpool->current_chunk, 1, memory_order_relaxed);
    }
#ifdef GGML_CPU_PROF
    if (mm_t0 != 0 && ith == 0) {
        fprintf(stderr, "[mm_prof] type=%-8s ne00=%lld ne01=%lld ne11=%lld total=%.0fus\n",
                ggml_type_name(src0->type), (long long) ne00, (long long) ne01, (long long) ne11,
                (double)(ggml_time_us() - mm_t0));
    }
#endif
}

// ggml_compute_forward_mul_mat_id

#define MMID_MATRIX_ROW(row_id, i1) matrix_rows[(row_id)*ids->ne[0]*ids->ne[1] + (i1)]

// INF-70 B3-k: flat (expert, row) slab partition of the single-token mul_mat_id — on by
// default; GGML_MMID_SLAB=0 restores D1's per-expert 1/nth row stripes (same-binary A/B).
// Same knob as the iqk dispatch hook, which is what GGML_IQK=1 actually runs.
// INF-70 GDN-ROWEXACT: batches of up to this many src1 rows are computed with the
// single-row kernels, one row at a time (bit-equal to single-token decode); GGML_ROWEXACT_N
// overrides the compiled default, 0 disables. Shared with the iqk dispatch (same env name).
// INF-70 BE-1: the compiled default is 0 (OFF). Row-exactness is a LOSSLESSNESS feature,
// not a speed feature: it costs 4-6% of decode on the MTP arms and buys no throughput, so
// it must not be switched on silently. Measured on the 24-prompt production mix:
// plain 12.484 (N=0) vs 12.516 (N=8) -- free on the trunk; MTP n-max 4 22.93 (N=0) vs
// 21.56 (N=8) -- a real cost. It is opt-in.
//
// CHOOSING A VALUE. ne11 is NOT the token count for every node. This model has 12 F32
// [128x64] mul_mat nodes per graph whose ne11 is 4*n_tokens (measured: ne11=4 on a
// single-token decode, ne11=12 on a 3-token batch; INF-70 batch-envelope dispatch trace).
// Two consequences, both counter-intuitive:
//   * any N >= 4 changes SINGLE-TOKEN decode numerics, because those nodes present ne11=4
//     at batch 1 -- so the served stream moves even with no batching anywhere;
//   * a value that covers the router (ne11 = n_tokens) can still MISS those nodes on the
//     verify batch. N=8 is exactly this trap: it catches ne11=4 at batch 1 but not the
//     ne11=20 of a 5-row verify batch, so it perturbs single decodes AND leaves the batch
//     non-row-exact. It is the worst of both and must not be used.
// For batch <= T rows to be row-exact, use N >= 4*T, i.e. N >= 4*(n_max+1) for MTP
// (n_max=4 -> N >= 20; 24 or 32 is a safe setting). Prefill is still never touched: a 512
// -row ubatch presents ne11=2048 at those nodes and ne11=512 at the rest.
#ifndef GGML_ROWEXACT_DEFAULT_N
#define GGML_ROWEXACT_DEFAULT_N 0
#endif
int ggml_cpu_rowexact_n(void) {
    static int n = -1;
    if (n < 0) {
        const char * s = getenv("GGML_ROWEXACT_N");
        n = s ? atoi(s) : GGML_ROWEXACT_DEFAULT_N;
        if (n < 0) n = 0;
    }
    return n;
}

static int ggml_mmid_slab_enabled(void) {
    static int flag = -1;
    if (flag < 0) {
        const char * s = getenv("GGML_MMID_SLAB");
        flag = (s == NULL || atoi(s) != 0) ? 1 : 0;
    }
    return flag;
}

struct mmid_row_mapping {
    int32_t i1;
    int32_t i2;
};

static void ggml_compute_forward_mul_mat_id_one_chunk(
    struct ggml_tensor * dst,
    const struct ggml_tensor * src0,
    const struct ggml_tensor * src1,
    const struct ggml_tensor * ids,
    const int64_t cur_a,
    const int64_t ir0_start,
    const int64_t ir0_end,
    const int64_t ir1_start,
    const int64_t ir1_end,
    const char * src0_cur,
    const struct mmid_row_mapping * matrix_rows,
    const size_t row_size,
    const bool src1_cont,
    const void * wdata) {

    GGML_TENSOR_BINARY_OP_LOCALS

    const enum ggml_type type = src0->type;

    ggml_vec_dot_t    const vec_dot      = type_traits_cpu[type].vec_dot;
    enum ggml_type    const vec_dot_type = type_traits_cpu[type].vec_dot_type;

    const int64_t blck_0 = 16;
    const int64_t blck_1 = 16;

    float tmp[16];

    for (int64_t iir1 = ir1_start; iir1 < ir1_end; iir1 += blck_1) {
        for (int64_t iir0 = ir0_start; iir0 < ir0_end; iir0 += blck_0) {
            for (int64_t ir1 = iir1; ir1 < iir1 + blck_1 && ir1 < ir1_end; ++ir1) {
                const int64_t _i12 = ir1; // logical row index for this expert

                struct mmid_row_mapping row_mapping = MMID_MATRIX_ROW(cur_a, _i12);
                const int id       = row_mapping.i1; // selected expert index

                const int64_t  i11 = id % ne11;
                const int64_t  i12 = row_mapping.i2; // row index in src1

                const int64_t  i1 = id;  // selected expert index
                const int64_t  i2 = i12; // row

                // desc: when src1 is not a contiguous memory block we have to calculate the offset using the strides
                //       if it is, then we have either copied the data to params->wdata and made it contiguous or we are using
                //       the original src1 data pointer, so we should index using the indices directly
                // TODO: this is a bit of a hack, we should probably have a better way to handle this
                const char * src1_col = (const char *) wdata +
                    (src1_cont || src1->type != vec_dot_type
                    ? (i11      + i12*ne11)*row_size
                    : (i11*nb11 + i12*nb12));

                float * dst_col = (float *) ((char *) dst->data + (i1*nb1 + i2*nb2));

                for (int64_t ir0 = iir0; ir0 < iir0 + blck_0 && ir0 < ir0_end; ++ir0) {
                    vec_dot(ne00, &tmp[ir0 - iir0], 0, src0_cur + ir0*nb01, 0, src1_col, 0, 1);
                }

                memcpy(&dst_col[iir0], tmp, (MIN(iir0 + blck_0, ir0_end) - iir0)*sizeof(float));
            }
        }
    }
}

static void * incr_ptr_aligned(void ** p, size_t size, size_t align) {

    void * ptr = *p;
    ptr = (void *) GGML_PAD((uintptr_t) ptr, align);
    *p = (void *) ((char *) ptr + size);
    return ptr;
}

static void ggml_compute_forward_mul_mat_id(
        const struct ggml_compute_params * params,
              struct ggml_tensor * dst) {

    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];
    const struct ggml_tensor * ids = dst->src[2];

    GGML_TENSOR_BINARY_OP_LOCALS

    const int ith = params->ith;
    const int nth = params->nth;

    const enum ggml_type type = src0->type;

    const bool src1_cont = ggml_is_contiguous(src1);

    enum ggml_type    const vec_dot_type    = type_traits_cpu[type].vec_dot_type;
    ggml_from_float_t const from_float      = type_traits_cpu[vec_dot_type].from_float;

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == ggml_type_size(type));
    GGML_ASSERT(nb10 == ggml_type_size(src1->type));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == sizeof(float));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

#if defined(GGML_USE_IQK_MULMAT)
    // iqk port (Stage 2): MoE expert GEMM fast path (ik_llama kernels). Runtime-gated by
    // env GGML_IQK=1; owns its own Q8_2_X4 src1 quantization + row-mapping + per-expert
    // iqk_mul_mat_moe. Returns true if handled (then we're done), false to fall through.
    if (ggml_iqk_try_mul_mat_id(params, dst)) {
        return;
    }
#endif

    // row groups
    const int n_ids = ids->ne[0]; // n_expert_used
    const int n_as  = ne02;       // n_expert

    // ---------------------------------------------------------------------------
    // INF-70 D1: single-token fast path — no internal barrier.
    //
    // At n_tokens == 1 the row->expert map is trivial (one row per used id), so every
    // thread builds it itself in a few dozen bytes of stack, quantizes the activations
    // into its own private slice of wdata, and iterates ONLY the used experts. That
    // deletes, per call: the serial grouping pass by thread 0, the n_as-entry (512 here)
    // scan by every thread, and the ggml_barrier that published them.
    //
    // Bit-exactness: the local insertion sort is stable and orders experts ascending,
    // reproducing exactly the (cur_a ascending, row ascending-by-id) iteration of the
    // shared path; the same from_float runs over the same whole rows; the chunk plan and
    // the vec_dot call order inside each chunk are unchanged. dst rows are disjoint per
    // id, so no accumulation is shared between experts.
    //
    // The per-expert chunk counter is only *read* when nth < nchunk0*nchunk1; the
    // eligibility test below rejects the fast path in that case, so the counter (whose
    // initialiser the barrier used to publish) is provably dead here.
    // ---------------------------------------------------------------------------
    #define GGML_MMID_B1_MAX_IDS 64

    bool mmid_batch1 = ids->ne[1] == 1 && ne13 == 1 && n_ids > 0 && n_ids <= GGML_MMID_B1_MAX_IDS;

    const size_t mmid_b1_region = ggml_row_size(vec_dot_type, ne10)*ne11*ne12*ne13;

    if (mmid_batch1 && src1->type != vec_dot_type && params->wsize < (size_t) nth * mmid_b1_region) {
        mmid_batch1 = false;
    }
    for (int64_t c = 1; mmid_batch1 && c <= n_ids; ++c) {
        int64_t c0, c1;
        ggml_mul_mat_chunk_plan(ne01, c, nth, &c0, &c1);
        if (nth < c0*c1) {
            mmid_batch1 = false;
        }
    }

    if (mmid_batch1) {
        const size_t row_size = ggml_row_size(vec_dot_type, ne10);

        const void * wdata_used = src1->data;

        if (src1->type != vec_dot_type) {
            GGML_ASSERT(src1->type == GGML_TYPE_F32);

            char * const wdata_priv = (char *) params->wdata + (size_t) ith * mmid_b1_region;

            const size_t nbw1 = row_size;
            const size_t nbw2 = nbw1*ne11;

            for (int64_t i12 = 0; i12 < ne12; ++i12) {
                for (int64_t i11 = 0; i11 < ne11; ++i11) {
                    from_float((const float *)((const char *) src1->data + i12*nb12 + i11*nb11),
                               (void *)               (wdata_priv      + i12*nbw2 + i11*nbw1),
                               ne10);
                }
            }

            wdata_used = wdata_priv;
        }

        // build the trivial row map locally, experts ascending, ids ascending within an expert
        int32_t                   b1_expert[GGML_MMID_B1_MAX_IDS];
        struct mmid_row_mapping   b1_row   [GGML_MMID_B1_MAX_IDS];
        int                       b1_n = 0;

        for (int id = 0; id < n_ids; ++id) {
            const int32_t i02 = *(const int32_t *) ((const char *) ids->data + id*ids->nb[0]);

            // Invalid expert IDs are inactive SER routes.  The output row is written by
            // nobody else, so a single thread may zero it without synchronisation.
            if (i02 < 0 || i02 >= n_as) {
                if (ith == 0) {
                    memset((char *) dst->data + id*nb1, 0, ne0*sizeof(float));
                }
                continue;
            }

            int pos = b1_n;
            while (pos > 0 && b1_expert[pos-1] > i02) {
                b1_expert[pos] = b1_expert[pos-1];
                b1_row   [pos] = b1_row   [pos-1];
                --pos;
            }
            b1_expert[pos] = i02;
            b1_row   [pos] = (struct mmid_row_mapping) { id, 0 };
            ++b1_n;
        }

        // ------------------------------------------------------------------------------
        // INF-70 B3-k: flat slab partition. The per-expert chunk plan below gives every
        // thread a 1/nth row stripe of EVERY used expert (14–54 rows, 19–26 KB, ten short
        // streams per op at 48 threads). Instead treat the used experts' rows as ONE flat
        // space of n_groups*ne01 rows (experts ascending — the order the loop below
        // iterates) and give thread ith the contiguous range [total*ith/nth,
        // total*(ith+1)/nth): one long run of at most two adjacent expert slabs, balanced
        // to ±1 row. Bit-identical: _one_chunk computes each output row with one vec_dot
        // over the whole row whatever the chunk bounds, and dst rows stay disjoint per
        // (id, row). Taken only for distinct experts (cne1 == 1 everywhere — the top-k
        // decode case); anything else keeps the per-expert chunk plan.
        // ------------------------------------------------------------------------------
        {
            int  grp_start[GGML_MMID_B1_MAX_IDS];
            int  n_groups = 0;
            bool slab_ok  = ggml_mmid_slab_enabled() && b1_n > 0;
            for (int i = 0; i < b1_n; ) {
                int j = i;
                while (j < b1_n && b1_expert[j] == b1_expert[i]) {
                    ++j;
                }
                if (j - i != 1) {
                    slab_ok = false;
                }
                grp_start[n_groups++] = i;
                i = j;
            }

            if (slab_ok) {
                const int64_t rows_total = (int64_t) n_groups * ne01;
                const int64_t r0 = (rows_total * ith) / nth;
                const int64_t r1 = (rows_total * (ith + 1)) / nth;

                for (int64_t r = r0; r < r1; ) {
                    const int     g  = (int) (r / ne01);
                    const int64_t g0 = (int64_t) g * ne01;
                    const int64_t re = MIN(r1, g0 + ne01);

                    const char * src0_cur = (const char *) src0->data + b1_expert[grp_start[g]]*nb02;

                    // cur_a == 0 with a base pointing at this expert's rows and at its
                    // single row-map entry reproduces MMID_MATRIX_ROW(cur_a, 0) exactly.
                    ggml_compute_forward_mul_mat_id_one_chunk(
                        dst, src0, src1, ids, /*cur_a =*/ 0,
                        r - g0, re - g0, /*ir1_start =*/ 0, /*ir1_end =*/ 1,
                        src0_cur, b1_row + grp_start[g], row_size, src1_cont, wdata_used
                    );

                    r = re;
                }

                return;
            }
        }

        for (int i = 0; i < b1_n; ) {
            int j = i;
            while (j < b1_n && b1_expert[j] == b1_expert[i]) {
                ++j;
            }

            const int64_t cur_a = b1_expert[i];
            const int64_t cne1  = j - i;

            const char * src0_cur = (const char *) src0->data + cur_a*nb02;

            int64_t nchunk0, nchunk1;
            ggml_mul_mat_chunk_plan(ne01, cne1, nth, &nchunk0, &nchunk1);
            GGML_ASSERT(nth >= nchunk0*nchunk1);

            const int64_t dr0 = (ne01 + nchunk0 - 1) / nchunk0;
            const int64_t dr1 = (cne1 + nchunk1 - 1) / nchunk1;

            if (ith < nchunk0*nchunk1) {
                const int64_t ith0 = ith % nchunk0;
                const int64_t ith1 = ith / nchunk0;

                const int64_t ir0_start = dr0 * ith0;
                const int64_t ir0_end   = MIN(ir0_start + dr0, ne01);

                const int64_t ir1_start = dr1 * ith1;
                const int64_t ir1_end   = MIN(ir1_start + dr1, cne1);

                // cur_a == 0 with a base pointing at this expert's rows reproduces
                // MMID_MATRIX_ROW(cur_a, i1) of the shared map exactly.
                ggml_compute_forward_mul_mat_id_one_chunk(
                    dst, src0, src1, ids, /*cur_a =*/ 0,
                    ir0_start, ir0_end, ir1_start, ir1_end,
                    src0_cur, b1_row + i, row_size, src1_cont, wdata_used
                );
            }

            i = j;
        }

        return;
    }

#ifdef GGML_CPU_PROF
    const int64_t mmid_t0 = ggml_cpu_prof_mm_enabled() ? ggml_time_us() : 0;
    int64_t mmid_t_quant = 0, mmid_t_map = 0, mmid_t_dots = 0;
#endif

    void * wdata_cur = params->wdata;

    if (src1->type != vec_dot_type) {
        incr_ptr_aligned(&wdata_cur, ggml_row_size(vec_dot_type, ggml_nelements(src1)), sizeof(int64_t));
    }

    int64_t * matrix_row_counts = // [n_as]
        incr_ptr_aligned(&wdata_cur, n_as*sizeof(int64_t), sizeof(int64_t));

    struct mmid_row_mapping * matrix_rows = // [n_as][ids->ne[0]*ids->ne[1]]
        incr_ptr_aligned(&wdata_cur, n_as*ids->ne[0]*ids->ne[1]*sizeof(struct mmid_row_mapping), sizeof(int64_t));

    char (*atomic_current_chunk)[CACHE_LINE_SIZE] = // [n_as]
        incr_ptr_aligned(&wdata_cur, CACHE_LINE_SIZE * n_as, CACHE_LINE_SIZE);

    GGML_ASSERT(params->wsize >= (size_t)((char *) wdata_cur - (char *) params->wdata));

    if (src1->type != vec_dot_type) {
        char * wdata = params->wdata;

        const size_t nbw0 = ggml_type_size(vec_dot_type);
        const size_t nbw1 = ggml_row_size(vec_dot_type, ne10);
        const size_t nbw2 = nbw1*ne11;
        const size_t nbw3 = nbw2*ne12;

        assert(params->wsize >= ne13*nbw3);
        GGML_ASSERT(src1->type == GGML_TYPE_F32);

#if 0
        for (int64_t i13 = 0; i13 < ne13; ++i13) {
            for (int64_t i12 = ith; i12 < ne12; i12 += nth) {
                for (int64_t i11 = 0; i11 < ne11; ++i11) {
                    from_float((float *)((char *) src1->data + i13*nb13 + i12*nb12 + i11*nb11),
                               (void *)               (wdata + i13*nbw3 + i12*nbw2 + i11*nbw1),
                               ne10);
                }
            }
        }
#else
        for (int64_t i13 = 0; i13 < ne13; ++i13) {
            for (int64_t i12 = 0; i12 < ne12; ++i12) {
                for (int64_t i11 = 0; i11 < ne11; ++i11) {
                    size_t bs = ggml_blck_size(vec_dot_type);
                    int64_t ne10_block_start = (ith * ne10/bs) / nth;
                    int64_t ne10_block_end   = ((ith + 1) * ne10/bs) / nth;
                    from_float((float *)((char *) src1->data + i13*nb13 + i12*nb12 + i11*nb11 + ne10_block_start*bs*nb10),
                               (void *)               (wdata + i13*nbw3 + i12*nbw2 + i11*nbw1 + ne10_block_start*nbw0),
                               (ne10_block_end - ne10_block_start) * bs);
                }
            }
        }
#endif
    }
#ifdef GGML_CPU_PROF
    if (mmid_t0 != 0) mmid_t_quant = ggml_time_us();
#endif

    if (ith == 0) {
        // initialize matrix_row_counts
        memset(matrix_row_counts, 0, n_as*sizeof(int64_t));

        // group rows by src0 matrix
        for (int64_t iid1 = 0; iid1 < ids->ne[1]; ++iid1) {
            for (int id = 0; id < n_ids; ++id) {
                const int32_t i02 = *(const int32_t *) ((const char *) ids->data + iid1*ids->nb[1] + id*ids->nb[0]);

                // Invalid expert IDs are inactive SER routes.  Do not rely on
                // assert here: NDEBUG builds must neither write outside the
                // row map nor leave the corresponding output row stale.
                if (i02 < 0 || i02 >= n_as) {
                    memset((char *) dst->data + id*nb1 + iid1*nb2, 0, ne0*sizeof(float));
                    continue;
                }

                MMID_MATRIX_ROW(i02, matrix_row_counts[i02]) = (struct mmid_row_mapping) {id, iid1};
                matrix_row_counts[i02] += 1;
            }
        }
    }

    // reset current_chunk
    for (int cur_a = ith; cur_a < n_as; cur_a += nth) {
        atomic_int * current_chunk_ctr = (atomic_int *)(atomic_current_chunk + cur_a);
        *current_chunk_ctr = nth;
    }

#ifdef GGML_CPU_PROF
    if (mmid_t0 != 0) mmid_t_map = ggml_time_us();
#endif
    ggml_barrier(params->threadpool);

    for (int cur_a = 0; cur_a < n_as; ++cur_a) {
        const int64_t cne1 = matrix_row_counts[cur_a];

        if (cne1 == 0) {
            continue;
        }

        const char * src0_cur = (const char *) src0->data + cur_a * nb02;
        const void * wdata = (src1->type == vec_dot_type) ? src1->data : params->wdata;
        const size_t row_size = ggml_row_size(vec_dot_type, ne10);

        const int64_t nr0 = ne01;
        const int64_t nr1 = cne1;

        int chunk_size = 16;
        if (nr0 == 1 || nr1 == 1) {
            chunk_size = 64;
        }

        // disable for NUMA
        const bool disable_chunking = ggml_is_numa();

        int64_t nchunk0 = (nr0 + chunk_size - 1) / chunk_size;
        int64_t nchunk1 = (nr1 + chunk_size - 1) / chunk_size;

        if (nchunk0 * nchunk1 < nth * 4 || disable_chunking) {
            nchunk0 = nr0 > nr1 ? nth : 1;
            nchunk1 = nr0 > nr1 ? 1 : nth;
        }

        const int64_t dr0 = (nr0 + nchunk0 - 1) / nchunk0;
        const int64_t dr1 = (nr1 + nchunk1 - 1) / nchunk1;

        int current_chunk = ith;

        atomic_int * current_chunk_ctr = (atomic_int *)(atomic_current_chunk + cur_a);

        while (current_chunk < nchunk0 * nchunk1) {
            const int64_t ith0 = current_chunk % nchunk0;
            const int64_t ith1 = current_chunk / nchunk0;

            const int64_t ir0_start = dr0 * ith0;
            const int64_t ir0_end = MIN(ir0_start + dr0, nr0);

            const int64_t ir1_start = dr1 * ith1;
            const int64_t ir1_end = MIN(ir1_start + dr1, nr1);

            ggml_compute_forward_mul_mat_id_one_chunk(
                dst, src0, src1, ids, cur_a,
                ir0_start, ir0_end, ir1_start, ir1_end,
                src0_cur, matrix_rows, row_size, src1_cont, wdata
            );

            if (nth >= nchunk0 * nchunk1) {
                break;
            }

            current_chunk = atomic_fetch_add_explicit(current_chunk_ctr, 1, memory_order_relaxed);
        }
    }
#ifdef GGML_CPU_PROF
    if (mmid_t0 != 0 && ith == 0) {
        mmid_t_dots = ggml_time_us();
        fprintf(stderr, "[mmid_prof] type=%-8s ne11=%lld n_as=%d quant=%.0fus map=%.0fus dots=%.0fus total=%.0fus\n",
                ggml_type_name(type), (long long) ne11, n_as,
                (double)(mmid_t_quant - mmid_t0), (double)(mmid_t_map - mmid_t_quant),
                (double)(mmid_t_dots - mmid_t_map), (double)(mmid_t_dots - mmid_t0));
    }
#endif
}

/////////////////////////////////

static void ggml_compute_forward(struct ggml_compute_params * params, struct ggml_tensor * tensor) {
    GGML_ASSERT(params);

    if (tensor->op == GGML_OP_NONE || ggml_is_empty(tensor)) {
        return;
    }

    // extra_buffer op?
    if (ggml_cpu_extra_compute_forward(params, tensor)) {
        return;
    }

    switch (tensor->op) {
        case GGML_OP_DUP:
            {
                ggml_compute_forward_dup(params, tensor);
            } break;
        case GGML_OP_ADD:
            {
                ggml_compute_forward_add(params, tensor);
            } break;
        case GGML_OP_ADD_ID:
            {
                ggml_compute_forward_add_id(params, tensor);
            } break;
        case GGML_OP_ADD1:
            {
                ggml_compute_forward_add1(params, tensor);
            } break;
        case GGML_OP_ACC:
            {
                ggml_compute_forward_acc(params, tensor);
            } break;
        case GGML_OP_SUB:
            {
                ggml_compute_forward_sub(params, tensor);
            } break;
        case GGML_OP_MUL:
            {
                ggml_compute_forward_mul(params, tensor);
            } break;
        case GGML_OP_DIV:
            {
                ggml_compute_forward_div(params, tensor);
            } break;
        case GGML_OP_SQR:
            {
                ggml_compute_forward_sqr(params, tensor);
            } break;
        case GGML_OP_SQRT:
            {
                ggml_compute_forward_sqrt(params, tensor);
            } break;
        case GGML_OP_LOG:
            {
                ggml_compute_forward_log(params, tensor);
            } break;
        case GGML_OP_SIN:
            {
                ggml_compute_forward_sin(params, tensor);
            } break;
        case GGML_OP_COS:
            {
                ggml_compute_forward_cos(params, tensor);
            } break;
        case GGML_OP_SUM:
            {
                ggml_compute_forward_sum(params, tensor);
            } break;
        case GGML_OP_SUM_ROWS:
            {
                ggml_compute_forward_sum_rows(params, tensor);
            } break;
        case GGML_OP_CUMSUM:
            {
                ggml_compute_forward_cumsum(params, tensor);
            } break;
        case GGML_OP_MEAN:
            {
                ggml_compute_forward_mean(params, tensor);
            } break;
        case GGML_OP_MEAN_D1:
            {
                ggml_compute_forward_mean_d1(params, tensor);
            } break;
        case GGML_OP_MOE_TOPK_NORM:
            {
                ggml_compute_forward_moe_topk_norm(params, tensor);
            } break;
        case GGML_OP_ARGMAX:
            {
                ggml_compute_forward_argmax(params, tensor);
            } break;
        case GGML_OP_COUNT_EQUAL:
            {
                ggml_compute_forward_count_equal(params, tensor);
            } break;
        case GGML_OP_REPEAT:
            {
                ggml_compute_forward_repeat(params, tensor);
            } break;
        case GGML_OP_REPEAT_BACK:
            {
                ggml_compute_forward_repeat_back(params, tensor);
            } break;
        case GGML_OP_CONCAT:
            {
                ggml_compute_forward_concat(params, tensor);
            } break;
        case GGML_OP_SILU_BACK:
            {
                ggml_compute_forward_silu_back(params, tensor);
            } break;
        case GGML_OP_NORM:
            {
                ggml_compute_forward_norm(params, tensor);
            } break;
        case GGML_OP_RMS_NORM:
            {
                ggml_compute_forward_rms_norm(params, tensor);
            } break;
        case GGML_OP_RMS_NORM_BACK:
            {
                ggml_compute_forward_rms_norm_back(params, tensor);
            } break;
        case GGML_OP_GROUP_NORM:
            {
                ggml_compute_forward_group_norm(params, tensor);
            } break;
        case GGML_OP_L2_NORM:
            {
                ggml_compute_forward_l2_norm(params, tensor);
            } break;
        case GGML_OP_MUL_MAT:
            {
                ggml_compute_forward_mul_mat(params, tensor);
            } break;
        case GGML_OP_MUL_MAT_ID:
            {
                ggml_compute_forward_mul_mat_id(params, tensor);
            } break;
        case GGML_OP_OUT_PROD:
            {
                ggml_compute_forward_out_prod(params, tensor);
            } break;
        case GGML_OP_SCALE:
            {
                ggml_compute_forward_scale(params, tensor);
            } break;
        case GGML_OP_SET:
            {
                ggml_compute_forward_set(params, tensor);
            } break;
        case GGML_OP_CPY:
            {
                ggml_compute_forward_cpy(params, tensor);
            } break;
        case GGML_OP_CONT:
            {
                ggml_compute_forward_cont(params, tensor);
            } break;
        case GGML_OP_GET_ROWS:
            {
                ggml_compute_forward_get_rows(params, tensor);
            } break;
        case GGML_OP_GET_ROWS_BACK:
            {
                ggml_compute_forward_get_rows_back(params, tensor);
            } break;
        case GGML_OP_SET_ROWS:
            {
                ggml_compute_forward_set_rows(params, tensor);
            } break;
        case GGML_OP_DIAG:
            {
                ggml_compute_forward_diag(params, tensor);
            } break;
        case GGML_OP_DIAG_MASK_INF:
            {
                ggml_compute_forward_diag_mask_inf(params, tensor);
            } break;
        case GGML_OP_DIAG_MASK_ZERO:
            {
                ggml_compute_forward_diag_mask_zero(params, tensor);
            } break;
        case GGML_OP_SOFT_MAX:
            {
                ggml_compute_forward_soft_max(params, tensor);
            } break;
        case GGML_OP_SOFT_MAX_BACK:
            {
                ggml_compute_forward_soft_max_ext_back(params, tensor);
            } break;
        case GGML_OP_ROPE:
            {
                ggml_compute_forward_rope(params, tensor);
            } break;
        case GGML_OP_ROPE_BACK:
            {
                ggml_compute_forward_rope_back(params, tensor);
            } break;
        case GGML_OP_CLAMP:
            {
                ggml_compute_forward_clamp(params, tensor);
            } break;
        case GGML_OP_CONV_TRANSPOSE_1D:
            {
                ggml_compute_forward_conv_transpose_1d(params, tensor);
            } break;
        case GGML_OP_IM2COL:
            {
                ggml_compute_forward_im2col(params, tensor);
            } break;
        case GGML_OP_IM2COL_BACK:
            {
                ggml_compute_forward_im2col_back_f32(params, tensor);
            } break;
        case GGML_OP_IM2COL_3D:
            {
                ggml_compute_forward_im2col_3d(params, tensor);
            } break;
        case GGML_OP_COL2IM_1D:
            {
                ggml_compute_forward_col2im_1d(params, tensor);
            } break;
        case GGML_OP_CONV_2D:
            {
                ggml_compute_forward_conv_2d(params, tensor);
            } break;
        case GGML_OP_CONV_3D:
            {
                ggml_compute_forward_conv_3d(params, tensor);
            } break;
        case GGML_OP_CONV_2D_DW:
            {
                ggml_compute_forward_conv_2d_dw(params, tensor);
            } break;
        case GGML_OP_CONV_TRANSPOSE_2D:
            {
                ggml_compute_forward_conv_transpose_2d(params, tensor);
            } break;
        case GGML_OP_POOL_1D:
            {
                ggml_compute_forward_pool_1d(params, tensor);
            } break;
        case GGML_OP_POOL_2D:
            {
                ggml_compute_forward_pool_2d(params, tensor);
            } break;
        case GGML_OP_POOL_2D_BACK:
            {
                ggml_compute_forward_pool_2d_back(params, tensor);
            } break;
        case GGML_OP_UPSCALE:
            {
                ggml_compute_forward_upscale(params, tensor);
            } break;
        case GGML_OP_PAD:
            {
                ggml_compute_forward_pad(params, tensor);
            } break;
        case GGML_OP_PAD_REFLECT_1D:
            {
                ggml_compute_forward_pad_reflect_1d(params, tensor);
            } break;
        case GGML_OP_ROLL:
            {
                ggml_compute_forward_roll(params, tensor);
            } break;
        case GGML_OP_ARANGE:
            {
                ggml_compute_forward_arange(params, tensor);
            } break;
        case GGML_OP_TIMESTEP_EMBEDDING:
            {
                ggml_compute_forward_timestep_embedding(params, tensor);
            } break;
        case GGML_OP_ARGSORT:
            {
                ggml_compute_forward_argsort(params, tensor);
            } break;
        case GGML_OP_TOP_K:
            {
                ggml_compute_forward_top_k(params, tensor);
            } break;
        case GGML_OP_LEAKY_RELU:
            {
                ggml_compute_forward_leaky_relu(params, tensor);
            } break;
        case GGML_OP_TRI:
            {
                ggml_compute_forward_tri(params, tensor);
            } break;
        case GGML_OP_FILL:
            {
                ggml_compute_forward_fill(params, tensor);
            } break;
        case GGML_OP_FLASH_ATTN_EXT:
            {
                ggml_compute_forward_flash_attn_ext(params, tensor);
            } break;
        case GGML_OP_FLASH_ATTN_BACK:
            {
                int32_t t = ggml_get_op_params_i32(tensor, 0);
                GGML_ASSERT(t == 0 || t == 1);
                bool masked = t != 0;
                ggml_compute_forward_flash_attn_back(params, masked, tensor);
            } break;
        case GGML_OP_SSM_CONV:
            {
                ggml_compute_forward_ssm_conv(params, tensor);
            } break;
        case GGML_OP_SSM_SCAN:
            {
                ggml_compute_forward_ssm_scan(params, tensor);
            } break;
        case GGML_OP_WIN_PART:
            {
                ggml_compute_forward_win_part(params, tensor);
            } break;
        case GGML_OP_WIN_UNPART:
            {
                ggml_compute_forward_win_unpart(params, tensor);
            } break;
        case GGML_OP_UNARY:
            {
                ggml_compute_forward_unary(params, tensor);
            } break;
        case GGML_OP_GLU:
            {
                ggml_compute_forward_glu(params, tensor);
            } break;
        case GGML_OP_GET_REL_POS:
            {
                ggml_compute_forward_get_rel_pos(params, tensor);
            } break;
        case GGML_OP_ADD_REL_POS:
            {
                ggml_compute_forward_add_rel_pos(params, tensor);
            } break;
        case GGML_OP_RWKV_WKV6:
            {
                ggml_compute_forward_rwkv_wkv6(params, tensor);
            } break;
        case GGML_OP_GATED_LINEAR_ATTN:
            {
                ggml_compute_forward_gla(params, tensor);
            } break;
        case GGML_OP_RWKV_WKV7:
            {
                ggml_compute_forward_rwkv_wkv7(params, tensor);
            } break;
        case GGML_OP_SOLVE_TRI:
            {
                ggml_compute_forward_solve_tri(params, tensor);
            } break;
        case GGML_OP_GATED_DELTA_NET:
            {
                ggml_compute_forward_gated_delta_net(params, tensor);
            } break;
        case GGML_OP_LIGHTNING_INDEXER:
            {
                ggml_compute_forward_lightning_indexer(params, tensor);
            } break;
        case GGML_OP_DSV4_HC_COMB:
            {
                ggml_compute_forward_dsv4_hc_comb(params, tensor);
            } break;
        case GGML_OP_DSV4_HC_PRE:
            {
                ggml_compute_forward_dsv4_hc_pre(params, tensor);
            } break;
        case GGML_OP_DSV4_HC_POST:
            {
                ggml_compute_forward_dsv4_hc_post(params, tensor);
            } break;
        case GGML_OP_MAP_CUSTOM1:
            {
                ggml_compute_forward_map_custom1(params, tensor);
            }
            break;
        case GGML_OP_MAP_CUSTOM2:
            {
                ggml_compute_forward_map_custom2(params, tensor);
            }
            break;
        case GGML_OP_MAP_CUSTOM3:
            {
                ggml_compute_forward_map_custom3(params, tensor);
            }
            break;
        case GGML_OP_CUSTOM:
            {
                ggml_compute_forward_custom(params, tensor);
            }
            break;
        case GGML_OP_CROSS_ENTROPY_LOSS:
            {
                ggml_compute_forward_cross_entropy_loss(params, tensor);
            }
            break;
        case GGML_OP_CROSS_ENTROPY_LOSS_BACK:
            {
                ggml_compute_forward_cross_entropy_loss_back(params, tensor);
            }
            break;
        case GGML_OP_OPT_STEP_ADAMW:
            {
                ggml_compute_forward_opt_step_adamw(params, tensor);
            }
            break;
        case GGML_OP_OPT_STEP_SGD:
            {
                ggml_compute_forward_opt_step_sgd(params, tensor);
            }
            break;
        case GGML_OP_NONE:
            {
                // nop
            } break;
        case GGML_OP_RESHAPE:
            {
                // nop
            } break;
        case GGML_OP_PERMUTE:
            {
                // nop
            } break;
        case GGML_OP_VIEW:
            {
                // nop
            } break;
        case GGML_OP_TRANSPOSE:
            {
                // nop
            } break;
        case GGML_OP_COUNT:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// Android's libc implementation "bionic" does not support setting affinity
#if defined(__gnu_linux__)
static void set_numa_thread_affinity(int thread_n) {
    if (!ggml_is_numa()) {
        return;
    }

    int node_num;
    int rv;
    size_t setsize = CPU_ALLOC_SIZE(g_state.numa.total_cpus);

    switch(g_state.numa.numa_strategy) {
        case GGML_NUMA_STRATEGY_DISTRIBUTE:
            // run thread on node_num thread_n / (threads per node)
            node_num = thread_n % g_state.numa.n_nodes;
            break;
        case GGML_NUMA_STRATEGY_ISOLATE:
            // run thread on current_node
            node_num = g_state.numa.current_node;
            break;
        case GGML_NUMA_STRATEGY_NUMACTL:
            // use the cpuset that numactl gave us
            rv = pthread_setaffinity_np(pthread_self(), setsize, &g_state.numa.cpuset);
            if (rv) {
                fprintf(stderr, "warning: pthread_setaffinity_np() failed: %s\n",strerror(rv));
            }
            return;
        default:
            return;
    }

    struct ggml_numa_node * node = &g_state.numa.nodes[node_num];

    cpu_set_t * cpus = CPU_ALLOC(g_state.numa.total_cpus);
    CPU_ZERO_S(setsize, cpus);
    for (size_t i = 0; i < node->n_cpus; ++i) {
        CPU_SET_S(node->cpus[i], setsize, cpus);
    }

    rv = pthread_setaffinity_np(pthread_self(), setsize, cpus);
    if (rv) {
            fprintf(stderr, "warning: pthread_setaffinity_np() failed: %s\n", strerror(rv));
    }

    CPU_FREE(cpus);
}

static void clear_numa_thread_affinity(void) {
    if (!ggml_is_numa()) {
        return;
    }

    size_t setsize = CPU_ALLOC_SIZE(g_state.numa.total_cpus);

    cpu_set_t * cpus = CPU_ALLOC(g_state.numa.total_cpus);
    CPU_ZERO_S(setsize, cpus);
    for (unsigned i = 0; i < g_state.numa.total_cpus; ++i) {
        CPU_SET_S(i, setsize, cpus);
    }

    int rv = pthread_setaffinity_np(pthread_self(), setsize, cpus);
    if (rv) {
        fprintf(stderr, "warning: pthread_setaffinity_np() failed: %s\n", strerror(rv));
    }

    CPU_FREE(cpus);
}
#else
// TODO: Windows etc.
// (the linux implementation may also work on BSD, someone should test)
static void set_numa_thread_affinity(int thread_n) { UNUSED(thread_n);  }
static void clear_numa_thread_affinity(void) {}
#endif

// minimum dst bytes for a multi-threaded get_rows; overridable for testing / tuning
static int64_t ggml_get_rows_min_bytes(void) {
    static int64_t v = -1;
    if (v < 0) {
        const char * s = getenv("GGML_GET_ROWS_MIN_BYTES");
        v = s ? atoll(s) : (64*1024);
        if (v < 0) {
            v = 0;
        }
    }
    return v;
}

static int ggml_get_n_tasks(struct ggml_tensor * node, int n_threads) {
    int n_tasks = 0;

    if (ggml_is_empty(node)) {
        // no need to multi-thread a no-op
        n_tasks = 1;
        return n_tasks;
    }

    switch (node->op) {
        case GGML_OP_CPY:
        case GGML_OP_DUP:
        case GGML_OP_CONT:
        case GGML_OP_ADD:
        case GGML_OP_ADD_ID:
        case GGML_OP_ADD1:
        case GGML_OP_ACC:
        case GGML_OP_CUMSUM:
        case GGML_OP_TRI:
        case GGML_OP_FILL:
            {
                n_tasks = n_threads;
            } break;
        case GGML_OP_SUB:
        case GGML_OP_SQR:
        case GGML_OP_SQRT:
        case GGML_OP_LOG:
        case GGML_OP_SIN:
        case GGML_OP_COS:
        case GGML_OP_SUM:
        case GGML_OP_SUM_ROWS:
        case GGML_OP_MEAN:
        case GGML_OP_ARGMAX:
            {
                n_tasks = 1;
            } break;
        case GGML_OP_MEAN_D1:
            {
                // parallel over ne0; give it real threads when there are rows to share
                n_tasks = MIN(n_threads, MAX(1, (int) node->src[0]->ne[0] / 32));
            } break;
        case GGML_OP_MOE_TOPK_NORM:
            {
                // parallel over tokens (ne1)
                n_tasks = MIN(n_threads, MAX(1, (int) node->src[0]->ne[1]));
            } break;
        case GGML_OP_COUNT_EQUAL:
        case GGML_OP_SOLVE_TRI:
        case GGML_OP_GATED_DELTA_NET:
        case GGML_OP_DSV4_HC_COMB:
        case GGML_OP_DSV4_HC_PRE:
        case GGML_OP_DSV4_HC_POST:
            {
                n_tasks = n_threads;
            } break;
        case GGML_OP_REPEAT:
        case GGML_OP_REPEAT_BACK:
        case GGML_OP_LEAKY_RELU:
            {
                n_tasks = 1;
            } break;
        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(node)) {
                case GGML_UNARY_OP_ABS:
                case GGML_UNARY_OP_SGN:
                case GGML_UNARY_OP_NEG:
                case GGML_UNARY_OP_STEP:
                case GGML_UNARY_OP_TANH:
                case GGML_UNARY_OP_ELU:
                case GGML_UNARY_OP_RELU:
                case GGML_UNARY_OP_SIGMOID:
                case GGML_UNARY_OP_HARDSWISH:
                case GGML_UNARY_OP_HARDSIGMOID:
                case GGML_UNARY_OP_EXP:
                case GGML_UNARY_OP_SOFTPLUS:
                case GGML_UNARY_OP_EXPM1:
                case GGML_UNARY_OP_FLOOR:
                case GGML_UNARY_OP_CEIL:
                case GGML_UNARY_OP_ROUND:
                case GGML_UNARY_OP_TRUNC:
                    {
                        n_tasks = 1;
                    } break;

                case GGML_UNARY_OP_GELU:
                case GGML_UNARY_OP_GELU_ERF:
                case GGML_UNARY_OP_GELU_QUICK:
                case GGML_UNARY_OP_SILU:
                case GGML_UNARY_OP_XIELU:
                    {
                        n_tasks = n_threads;
                    } break;
                default:
                    GGML_ABORT("fatal error");
            }
            break;
        case GGML_OP_GLU:
            switch (ggml_get_glu_op(node)) {
                case GGML_GLU_OP_REGLU:
                case GGML_GLU_OP_GEGLU:
                case GGML_GLU_OP_SWIGLU:
                case GGML_GLU_OP_SWIGLU_OAI:
                case GGML_GLU_OP_GEGLU_ERF:
                case GGML_GLU_OP_GEGLU_QUICK:
                    {
                        n_tasks = n_threads;
                    } break;
                default:
                    GGML_ABORT("fatal error");
            }
            break;
        case GGML_OP_SILU_BACK:
        case GGML_OP_MUL:
        case GGML_OP_DIV:
        case GGML_OP_NORM:
        case GGML_OP_RMS_NORM:
        case GGML_OP_RMS_NORM_BACK:
        case GGML_OP_L2_NORM:
        case GGML_OP_GROUP_NORM:
        case GGML_OP_CONCAT:
        case GGML_OP_MUL_MAT:
        case GGML_OP_MUL_MAT_ID:
        case GGML_OP_OUT_PROD:
            {
                n_tasks = n_threads;
            } break;
        case GGML_OP_GET_ROWS:
            {
                // the CPU get_rows kernels split the work over (row, column-chunk) pairs, so they
                // are correct for any n_tasks and bit-identical to the single-threaded result.
                // Small gathers stay single-task: below ~64 KB of output the barrier and the
                // thread wake-up cost more than the copy (this is what the old FIXME was about).
                // INF-70 D8: 175 GET_ROWS/token cost 9.34 ms on one thread, 8.9 ms of it in 72
                // nodes that gather a single 3 MB / 120 KB f32 row.
                n_tasks = ggml_nbytes(node) >= ggml_get_rows_min_bytes() ? n_threads : 1;
            } break;
        case GGML_OP_SET_ROWS:
            {
                // NOT parallelised: set_rows splits over source rows, but two source rows may
                // carry the SAME destination index, so threads are not provably disjoint on the
                // destination. Measured cost is 0.011 ms/token (INF-70 D0) - nothing to win.
                n_tasks = 1;
            } break;
        case GGML_OP_SCALE:
        case GGML_OP_SET:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
        case GGML_OP_GET_ROWS_BACK:
        case GGML_OP_DIAG:
            {
                n_tasks = 1;
            } break;
        case GGML_OP_DIAG_MASK_ZERO:
        case GGML_OP_DIAG_MASK_INF:
        case GGML_OP_SOFT_MAX_BACK:
        case GGML_OP_ROPE:
        case GGML_OP_ROPE_BACK:
        case GGML_OP_ADD_REL_POS:
            {
                n_tasks = n_threads;
            } break;
        case GGML_OP_CLAMP:
            {
                n_tasks = 1; //TODO
            } break;
        case GGML_OP_SOFT_MAX:
            {
                n_tasks = MIN(n_threads, ggml_nrows(node->src[0]));
            } break;
        case GGML_OP_IM2COL:
        case GGML_OP_IM2COL_BACK:
        case GGML_OP_IM2COL_3D:
        case GGML_OP_CONV_2D:
        case GGML_OP_CONV_3D:
        case GGML_OP_CONV_2D_DW:
        case GGML_OP_COL2IM_1D:
        case GGML_OP_CONV_TRANSPOSE_1D:
        case GGML_OP_CONV_TRANSPOSE_2D:
            {
                n_tasks = n_threads;
            } break;
        case GGML_OP_POOL_1D:
        case GGML_OP_POOL_2D:
        case GGML_OP_POOL_2D_BACK:
            {
                n_tasks = 1;
            } break;
        case GGML_OP_UPSCALE:
        case GGML_OP_PAD:
        case GGML_OP_PAD_REFLECT_1D:
        case GGML_OP_ROLL:
        case GGML_OP_ARANGE:
        case GGML_OP_TIMESTEP_EMBEDDING:
        case GGML_OP_ARGSORT:
        case GGML_OP_TOP_K:
        case GGML_OP_FLASH_ATTN_EXT:
        case GGML_OP_FLASH_ATTN_BACK:
        case GGML_OP_SSM_CONV:
        case GGML_OP_SSM_SCAN:
        case GGML_OP_LIGHTNING_INDEXER:
            {
                n_tasks = n_threads;
            } break;
        case GGML_OP_RWKV_WKV6:
        case GGML_OP_GATED_LINEAR_ATTN:
        case GGML_OP_RWKV_WKV7:
            {
                const int64_t n_heads = node->src[1]->ne[1];
                n_tasks = MIN(n_threads, n_heads);
            } break;
        case GGML_OP_WIN_PART:
        case GGML_OP_WIN_UNPART:
        case GGML_OP_GET_REL_POS:
            {
                n_tasks = 1;
            } break;
        case GGML_OP_MAP_CUSTOM1:
            {
                struct ggml_map_custom1_op_params p;
                memcpy(&p, node->op_params, sizeof(p));
                if (p.n_tasks == GGML_N_TASKS_MAX) {
                    n_tasks = n_threads;
                } else {
                    n_tasks = MIN(p.n_tasks, n_threads);
                }
            } break;
        case GGML_OP_MAP_CUSTOM2:
            {
                struct ggml_map_custom2_op_params p;
                memcpy(&p, node->op_params, sizeof(p));
                if (p.n_tasks == GGML_N_TASKS_MAX) {
                    n_tasks = n_threads;
                } else {
                    n_tasks = MIN(p.n_tasks, n_threads);
                }
            } break;
        case GGML_OP_MAP_CUSTOM3:
            {
                struct ggml_map_custom3_op_params p;
                memcpy(&p, node->op_params, sizeof(p));
                if (p.n_tasks == GGML_N_TASKS_MAX) {
                    n_tasks = n_threads;
                } else {
                    n_tasks = MIN(p.n_tasks, n_threads);
                }
            } break;
        case GGML_OP_CUSTOM:
            {
                struct ggml_custom_op_params p;
                memcpy(&p, node->op_params, sizeof(p));
                if (p.n_tasks == GGML_N_TASKS_MAX) {
                    n_tasks = n_threads;
                } else {
                    n_tasks = MIN(p.n_tasks, n_threads);
                }
            } break;
        case GGML_OP_CROSS_ENTROPY_LOSS:
        case GGML_OP_CROSS_ENTROPY_LOSS_BACK:
        case GGML_OP_OPT_STEP_ADAMW:
        case GGML_OP_OPT_STEP_SGD:
            {
                n_tasks = n_threads;
            } break;
        case GGML_OP_NONE:
            {
                n_tasks = 1;
            } break;
        case GGML_OP_COUNT:
            {
                GGML_ABORT("fatal error");
            }
        default:
            {
                fprintf(stderr, "%s: op not implemented: ", __func__);
                if (node->op < GGML_OP_COUNT) {
                    fprintf(stderr, "%s\n", ggml_op_name(node->op));
                } else {
                    fprintf(stderr, "%d\n", node->op);
                }
                GGML_ABORT("fatal error");
            }
    }

    assert(n_tasks > 0);

    return n_tasks;
}

static thread_ret_t ggml_graph_compute_secondary_thread(void* data);

#if defined(_WIN32)
#include "windows.h"

// TODO: support > 64 CPUs
static bool ggml_thread_apply_affinity(bool * mask) {
    HANDLE    h = GetCurrentThread();
    uint64_t  bitmask = 0ULL;

    assert(GGML_MAX_N_THREADS >= 64);

    for (int32_t i = 0; i < 8; i++) {
        int32_t idx = i * 8;
        uint8_t val = 0;
        val |= mask[idx + 0] << 0;
        val |= mask[idx + 1] << 1;
        val |= mask[idx + 2] << 2;
        val |= mask[idx + 3] << 3;
        val |= mask[idx + 4] << 4;
        val |= mask[idx + 5] << 5;
        val |= mask[idx + 6] << 6;
        val |= mask[idx + 7] << 7;
        bitmask |= (uint64_t)val << idx;
    }

    for (int32_t i = 64; i < GGML_MAX_N_THREADS; i++) {
        if (mask[i]) {
            fprintf(stderr, "warn: setting thread-affinity for > 64 CPUs isn't supported on windows!\n");
            break;
        }
    }

    DWORD_PTR m = (DWORD_PTR)bitmask;

    m = SetThreadAffinityMask(h, m);

    return m != 0;
}

static bool ggml_thread_apply_priority(int32_t prio) {
    // Note that on Windows the Process Priority Class must be updated in order to set Thread priority.
    // This is up to the applications.
    DWORD p = THREAD_PRIORITY_NORMAL;
    switch (prio) {
        case GGML_SCHED_PRIO_LOW:      p = THREAD_PRIORITY_BELOW_NORMAL;  break;
        case GGML_SCHED_PRIO_NORMAL:   p = THREAD_PRIORITY_NORMAL;        break;
        case GGML_SCHED_PRIO_MEDIUM:   p = THREAD_PRIORITY_ABOVE_NORMAL;  break;
        case GGML_SCHED_PRIO_HIGH:     p = THREAD_PRIORITY_HIGHEST;       break;
        case GGML_SCHED_PRIO_REALTIME: p = THREAD_PRIORITY_TIME_CRITICAL; break;
    }

    if (prio != GGML_SCHED_PRIO_LOW) {
        // Tell Windows that this thread should not be throttled (needs its own CPU core).
        // Newer Windows 11 versions aggressively park (offline) CPU cores and often place
        // all our threads onto the first 4 cores which results in terrible performance with
        // n_threads > 4
        #if _WIN32_WINNT >= 0x0602
        THREAD_POWER_THROTTLING_STATE t;
        ZeroMemory(&t, sizeof(t));
        t.Version     = THREAD_POWER_THROTTLING_CURRENT_VERSION;
        t.ControlMask = THREAD_POWER_THROTTLING_EXECUTION_SPEED;
        t.StateMask   = 0;

        if (!SetThreadInformation(GetCurrentThread(), ThreadPowerThrottling, &t, sizeof(t))) {
            GGML_LOG_DEBUG("failed to disable thread power throttling %d : (%d)\n", prio, (int) GetLastError());
            return false;
        }
        #endif
    }

    if (prio == GGML_SCHED_PRIO_NORMAL) {
        // Keep inherited policy/priority
        return true;
    }

    if (!SetThreadPriority(GetCurrentThread(), p)) {
        fprintf(stderr, "warn: failed to set thread priority %d : (%d)\n", prio, (int) GetLastError());
        return false;
    }

    return true;
}

#elif defined(__APPLE__)
#include <sys/types.h>
#include <sys/resource.h>

static bool ggml_thread_apply_affinity(const bool * mask) {
    // Not supported on Apple platforms
    UNUSED(mask);
    return true;
}

static bool ggml_thread_apply_priority(int32_t prio) {
    struct sched_param p;
    int32_t policy = SCHED_OTHER;
    switch (prio) {
        // TODO: there seems to be no way to set lower prio on Apple platforms
        case GGML_SCHED_PRIO_LOW:      policy = SCHED_OTHER; p.sched_priority = 0;  break;
        case GGML_SCHED_PRIO_NORMAL:   policy = SCHED_OTHER; p.sched_priority = 0;  break;
        case GGML_SCHED_PRIO_MEDIUM:   policy = SCHED_FIFO;  p.sched_priority = 40; break;
        case GGML_SCHED_PRIO_HIGH:     policy = SCHED_FIFO;  p.sched_priority = 80; break;
        case GGML_SCHED_PRIO_REALTIME: policy = SCHED_FIFO;  p.sched_priority = 90; break;
    }

    if (prio == GGML_SCHED_PRIO_NORMAL) {
        // Keep inherited policy/priority
        return true;
    }

    int32_t err = pthread_setschedparam(pthread_self(), policy, &p);
    if (err != 0) {
        fprintf(stderr, "warn: failed to set thread priority %d : %s (%d)\n", prio, strerror(err), err);
        return false;
    }

    return true;
}

#elif defined(__gnu_linux__)
// TODO: this may not work on BSD, to be verified

static bool ggml_thread_apply_affinity(const bool * mask) {
    cpu_set_t cpuset;
    int err;

    CPU_ZERO(&cpuset);

    for (uint32_t i = 0; i < GGML_MAX_N_THREADS; i++) {
        if (mask[i]) {
            GGML_PRINT_DEBUG("Thread %lx: adding %d to cpuset\n", pthread_self(), i);
            CPU_SET(i, &cpuset);
        }
    }

#ifdef __ANDROID__
    err = sched_setaffinity(0, sizeof(cpuset), &cpuset);
    if (err < 0) {
        err = errno;
    }
#else
    err = pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset);
#endif
    if (err != 0) {
        fprintf(stderr, "warn: failed to set affinity mask 0x%llx : %s (%d)\n", (unsigned long long)mask, strerror(err), err);
        return false;
    }

    return true;
}

static bool ggml_thread_apply_priority(int32_t prio) {
    struct sched_param p;
    int32_t policy = SCHED_OTHER;
    switch (prio) {
        case GGML_SCHED_PRIO_LOW:      policy = SCHED_BATCH; p.sched_priority = 0;  break;
        case GGML_SCHED_PRIO_NORMAL:   policy = SCHED_OTHER; p.sched_priority = 0;  break;
        case GGML_SCHED_PRIO_MEDIUM:   policy = SCHED_FIFO;  p.sched_priority = 40; break;
        case GGML_SCHED_PRIO_HIGH:     policy = SCHED_FIFO;  p.sched_priority = 80; break;
        case GGML_SCHED_PRIO_REALTIME: policy = SCHED_FIFO;  p.sched_priority = 90; break;
    }

    if (prio == GGML_SCHED_PRIO_NORMAL) {
        // Keep inherited policy/priority
        return true;
    }

    int32_t err = pthread_setschedparam(pthread_self(), policy, &p);
    if (err != 0) {
        fprintf(stderr, "warn: failed to set thread priority %d : %s (%d)\n", prio, strerror(err), err);
        return false;
    }

    return true;
}

#else // unsupported platforms

static bool ggml_thread_apply_affinity(const bool * mask) {
    UNUSED(mask);
    return true;
}

static bool ggml_thread_apply_priority(int32_t prio) {
    UNUSED(prio);
    return true;
}

#endif

static bool ggml_thread_cpumask_is_valid(const bool * mask) {
    for (int i = 0; i < GGML_MAX_N_THREADS; i++) {
        if (mask[i]) { return true; }
    }
    return false;
}

static void ggml_thread_cpumask_next(const bool * global_mask, bool * local_mask, bool strict, int32_t* iter) {
    if (!strict) {
        memcpy(local_mask, global_mask, GGML_MAX_N_THREADS);
        return;
    } else {
        memset(local_mask, 0, GGML_MAX_N_THREADS);
        int32_t base_idx = *iter;
        for (int32_t i = 0; i < GGML_MAX_N_THREADS; i++) {
            int32_t idx = base_idx + i;
            if (idx >= GGML_MAX_N_THREADS) {
                // Just a cheaper modulo
                idx -= GGML_MAX_N_THREADS;
            }
            if (global_mask[idx]) {
                local_mask[idx] = 1;
                *iter = idx + 1;
                return;
            }
        }
    }
}

void ggml_threadpool_free(struct ggml_threadpool* threadpool) {
    if (!threadpool) return;

    const int n_threads = threadpool->n_threads;

#ifndef GGML_USE_OPENMP
    struct ggml_compute_state* workers = threadpool->workers;

    ggml_mutex_lock(&threadpool->mutex);

    threadpool->stop = true;
    threadpool->pause = false;

    ggml_cond_broadcast(&threadpool->cond);
    ggml_mutex_unlock(&threadpool->mutex);

    for (int j = 1; j < n_threads; j++) {
        int32_t rc = ggml_thread_join(workers[j].thrd, NULL);
        GGML_ASSERT(rc == GGML_EXIT_SUCCESS || rc == GGML_EXIT_ABORTED);
        UNUSED(rc);
    }

    ggml_mutex_destroy(&threadpool->mutex);
    ggml_cond_destroy(&threadpool->cond);
#endif // GGML_USE_OPENMP

    const size_t workers_size = sizeof(struct ggml_compute_state) * n_threads;
    ggml_aligned_free(threadpool->workers, workers_size);
    ggml_aligned_free(threadpool, sizeof(struct ggml_threadpool));
}

#ifndef GGML_USE_OPENMP
// pause/resume must be called under mutex
static void ggml_threadpool_pause_locked(struct ggml_threadpool * threadpool) {
    GGML_PRINT_DEBUG("Pausing threadpool\n");
    threadpool->pause = true;
    ggml_cond_broadcast(&threadpool->cond);
}

static void ggml_threadpool_resume_locked(struct ggml_threadpool * threadpool) {
    GGML_PRINT_DEBUG("Resuming threadpool\n");
    threadpool->pause = false;
    ggml_cond_broadcast(&threadpool->cond);
}
#endif

void ggml_threadpool_pause(struct ggml_threadpool * threadpool) {
#ifndef GGML_USE_OPENMP
    ggml_mutex_lock(&threadpool->mutex);
    if (!threadpool->pause) {
       ggml_threadpool_pause_locked(threadpool);
    }
    ggml_mutex_unlock(&threadpool->mutex);
#else
    UNUSED(threadpool);
#endif
}

void ggml_threadpool_resume(struct ggml_threadpool * threadpool) {
#ifndef GGML_USE_OPENMP
    ggml_mutex_lock(&threadpool->mutex);
    if (threadpool->pause) {
       ggml_threadpool_resume_locked(threadpool);
    }
    ggml_mutex_unlock(&threadpool->mutex);
#else
    UNUSED(threadpool);
#endif
}

struct ggml_cplan ggml_graph_plan(
          const struct ggml_cgraph * cgraph,
                               int   n_threads,
            struct ggml_threadpool * threadpool) {

    if (threadpool == NULL) {
        //GGML_PRINT_DEBUG("Threadpool is not specified. Will create a disposable threadpool : n_threads %d\n", n_threads);
    }
    if (n_threads <= 0) {
        n_threads = threadpool ? threadpool->n_threads : GGML_DEFAULT_N_THREADS;
    }

#if defined(__EMSCRIPTEN__) && !defined(__EMSCRIPTEN_PTHREADS__)
    // Emscripten without pthreads support can only use a single thread
    n_threads = 1;
#endif

    size_t work_size = 0;

    struct ggml_cplan cplan;
    memset(&cplan, 0, sizeof(struct ggml_cplan));

    int max_tasks = 1;

    // thread scheduling for the different operations + work buffer size estimation
    for (int i = 0; i < cgraph->n_nodes; i++) {
        struct ggml_tensor * node = cgraph->nodes[i];

        const int n_tasks = ggml_get_n_tasks(node, n_threads);

        max_tasks = MAX(max_tasks, n_tasks);

        size_t cur = 0;

        if (!ggml_cpu_extra_work_size(n_threads, node, &cur)) {
            switch (node->op) {
                case GGML_OP_CPY:
                case GGML_OP_DUP:
                    {
                        if (ggml_is_quantized(node->type) ||
                            // F16 -> BF16 and BF16 -> F16 copies go through intermediate F32
                            (node->src[0]->type == GGML_TYPE_F16  && node->src[1] && node->src[1]->type == GGML_TYPE_BF16) ||
                            (node->src[0]->type == GGML_TYPE_BF16 && node->src[1] && node->src[1]->type == GGML_TYPE_F16) ||
                            // conversion between F32 and I32
                            (node->src[0]->type == GGML_TYPE_F32 && node->src[1] && node->src[1]->type == GGML_TYPE_I32) ||
                            (node->src[0]->type == GGML_TYPE_I32 && node->src[1] && node->src[1]->type == GGML_TYPE_F32)) {
                            cur = ggml_type_size(GGML_TYPE_F32) * node->ne[0] * n_tasks;
                        }
                    } break;
                case GGML_OP_ADD:
                case GGML_OP_ADD_ID:
                case GGML_OP_ADD1:
                    {
                        if (ggml_is_quantized(node->src[0]->type)) {
                            cur = ggml_type_size(GGML_TYPE_F32) * node->src[0]->ne[0] * n_tasks;
                        }
                    } break;
                case GGML_OP_ACC:
                    {
                        if (ggml_is_quantized(node->src[0]->type)) {
                            cur = ggml_type_size(GGML_TYPE_F32) * node->src[1]->ne[0] * n_tasks;
                        }
                    } break;
                case GGML_OP_COUNT_EQUAL:
                    {
                        cur = ggml_type_size(node->type)*n_tasks;
                    } break;
                case GGML_OP_MUL_MAT:
                    {
                        const enum ggml_type vec_dot_type = type_traits_cpu[node->src[0]->type].vec_dot_type;

                        if (node->src[1]->type != vec_dot_type) {
                            cur = ggml_row_size(vec_dot_type, ggml_nelements(node->src[1]));
                            // INF-70 D1: at a single src1 row every thread quantizes the whole row
                            // into its own private slice (no internal barrier), so the buffer holds
                            // n_tasks copies of it. This is a superset of the runtime predicate:
                            // the forward falls back to the shared path if wsize is short.
                            if (node->src[1]->ne[1]*node->src[1]->ne[2]*node->src[1]->ne[3] == 1) {
                                cur *= n_tasks;
                            }
                        }
                    } break;
                case GGML_OP_MUL_MAT_ID:
                    {
                        cur = 0;
                        const struct ggml_tensor * src0 = node->src[0];
                        const struct ggml_tensor * src1 = node->src[1];
                        const struct ggml_tensor * ids = node->src[2];
                        const enum ggml_type vec_dot_type = type_traits_cpu[src0->type].vec_dot_type;
                        const int n_as = src0->ne[2];
                        // src1
                        if (src1->type != vec_dot_type) {
                            cur += ggml_row_size(vec_dot_type, ggml_nelements(src1)) + sizeof(int64_t);
                            // INF-70 D1: at a single token every thread quantizes the activations
                            // into its own private slice (no internal barrier). Superset of the
                            // runtime predicate: the forward falls back if wsize is short.
                            if (ids->ne[1] == 1) {
                                cur += (size_t) (n_tasks - 1) * ggml_row_size(vec_dot_type, ggml_nelements(src1));
                            }
                        }
                        // matrix_row_counts
                        cur += n_as * sizeof(int64_t) + sizeof(int64_t);
                        // matrix_rows
                        cur += n_as*ids->ne[0]*ids->ne[1]*sizeof(struct mmid_row_mapping) + sizeof(int64_t);
                        // atomic_current_chunk
                        cur += CACHE_LINE_SIZE*n_as + CACHE_LINE_SIZE;
                    } break;
                case GGML_OP_OUT_PROD:
                    {
                        if (ggml_is_quantized(node->src[0]->type) ||
                            node->src[0]->type == GGML_TYPE_F16) {
                            cur = ggml_type_size(GGML_TYPE_F32) * node->src[0]->ne[0] * n_tasks;
                        }
                    } break;
                case GGML_OP_SET_ROWS:
                    {
                        if (node->src[0]->type == GGML_TYPE_F16 && node->type != GGML_TYPE_F16) {
                            cur = ggml_type_size(GGML_TYPE_F32) * node->src[0]->ne[0] * n_tasks;
                        }
                    } break;
                case GGML_OP_SOFT_MAX:
                case GGML_OP_ROPE:
                case GGML_OP_ROPE_BACK:
                    {
                        cur = ggml_type_size(GGML_TYPE_F32) * node->ne[0] * n_tasks;
                    } break;
                case GGML_OP_CONV_TRANSPOSE_1D:
                    {
                        GGML_ASSERT(node->src[0]->ne[3] == 1);
                        GGML_ASSERT(node->src[1]->ne[2] == 1);
                        GGML_ASSERT(node->src[1]->ne[3] == 1);

                        const int64_t ne00 = node->src[0]->ne[0];  // K
                        const int64_t ne01 = node->src[0]->ne[1];  // Cout
                        const int64_t ne02 = node->src[0]->ne[2];  // Cin
                        const int64_t ne10 = node->src[1]->ne[0];  // L
                        const int64_t ne11 = node->src[1]->ne[1];  // Cin

                        if ((node->src[0]->type == GGML_TYPE_F16 ||
                             node->src[0]->type == GGML_TYPE_BF16) &&
                            node->src[1]->type == GGML_TYPE_F32) {
                            cur += sizeof(ggml_fp16_t)*ne00*ne01*ne02;
                            cur += sizeof(ggml_fp16_t)*ne10*ne11;
                        } else if (node->src[0]->type == GGML_TYPE_F32 &&
                                   node->src[1]->type == GGML_TYPE_F32) {
                            cur += sizeof(float)*ne00*ne01*ne02;
                            cur += sizeof(float)*ne10*ne11;
                        } else {
                            GGML_ABORT("fatal error");
                        }
                    } break;
                case GGML_OP_CONV_2D:
                case GGML_OP_CONV_3D:
                    {
                        cur = GGML_IM2COL_WORK_SIZE;
                    } break;
                case GGML_OP_CONV_TRANSPOSE_2D:
                    {
                        const int64_t ne00 = node->src[0]->ne[0]; // W
                        const int64_t ne01 = node->src[0]->ne[1]; // H
                        const int64_t ne02 = node->src[0]->ne[2]; // Channels Out
                        const int64_t ne03 = node->src[0]->ne[3]; // Channels In

                        const int64_t ne10 = node->src[1]->ne[0]; // W
                        const int64_t ne11 = node->src[1]->ne[1]; // H
                        const int64_t ne12 = node->src[1]->ne[2]; // Channels In

                        GGML_ASSERT(node->src[0]->type == GGML_TYPE_F16 || node->src[0]->type == GGML_TYPE_F32);
                        GGML_ASSERT(node->src[1]->type == GGML_TYPE_F32);

                        cur += ggml_type_size(node->src[0]->type) * ne00 * ne01 * ne02 * ne03;
                        cur += ggml_type_size(node->src[0]->type) * ne10 * ne11 * ne12;

                    } break;
                case GGML_OP_TOP_K:
                    {
                        cur += sizeof(int32_t)*node->src[0]->ne[0]*n_tasks;
                    } break;
                case GGML_OP_FLASH_ATTN_EXT:
                    {
                        const int64_t neq2 = node->src[0]->ne[2]; // number of query heads
                        const int64_t DK = node->src[1]->ne[0];
                        const int64_t DV = node->src[2]->ne[0];

                        // Tiled flash attention scratch (tile sizes defined in common.h)
                        // Per-thread: Q_q + KQ + mask + VKQ32 + V32 + K_f32 + padding
                        size_t prefill  = sizeof(float)*(GGML_FA_TILE_Q*DK + 2*GGML_FA_TILE_Q*GGML_FA_TILE_KV + GGML_FA_TILE_Q*DV + GGML_FA_TILE_KV*DV + GGML_FA_TILE_KV*DK)*n_tasks;

                        // Decode path: n_kv_chunks = n_tasks (one chunk per thread)
                        // Per-thread: VKQ accmulator (DV), partial M, partial S + intra-thread scratch for V, Q and VKQ
                        size_t n_chunks = n_tasks;
                        size_t decode   = sizeof(float)*(neq2*n_chunks*(2+DV) + n_tasks*(DK + 2*DV));

                        cur += MAX(prefill, decode);
                    } break;
                case GGML_OP_FLASH_ATTN_BACK:
                    {
                        const int64_t    D = node->src[0]->ne[0];
                        const int64_t ne11 = ggml_up(node->src[1]->ne[1], GGML_SOFT_MAX_UNROLL);
                        const int64_t mxDn = MAX(D, ne11) * 2; // *2 because of S and SM in ggml_compute_forward_flash_attn_back
                        if (node->src[1]->type == GGML_TYPE_F32) {
                            cur  = sizeof(float)*mxDn*n_tasks; // TODO: this can become (n_tasks-1)
                            cur += sizeof(float)*mxDn*n_tasks; // this is overestimated by x2
                        } else if (node->src[1]->type == GGML_TYPE_F16) {
                            cur  = sizeof(float)*mxDn*n_tasks; // TODO: this can become (n_tasks-1)
                            cur += sizeof(float)*mxDn*n_tasks; // this is overestimated by x2
                        } else if (node->src[1]->type == GGML_TYPE_BF16) {
                            cur  = sizeof(float)*mxDn*n_tasks; // TODO: this can become (n_tasks-1)
                            cur += sizeof(float)*mxDn*n_tasks; // this is overestimated by x2
                        }
                    } break;

                case GGML_OP_CROSS_ENTROPY_LOSS:
                    {
                        cur = ggml_type_size(node->type)*(n_tasks + node->src[0]->ne[0]*n_tasks);
                    } break;
                case GGML_OP_GATED_DELTA_NET:
                    {
                        const int64_t S_v = node->src[2]->ne[0];
                        const int64_t K   = ggml_get_op_params_i32(node, 0);
                        const int64_t per_thread = S_v + (K > 1 ? S_v * S_v : 0);
                        cur = per_thread * sizeof(float) * n_tasks;
                    } break;
                case GGML_OP_COUNT:
                    {
                        GGML_ABORT("fatal error");
                    }
                case GGML_OP_LIGHTNING_INDEXER:
                    {
                        // temp buffer for dequantizing lightning indexer keys
                        const int64_t ne10 = node->src[1]->ne[0];
                        cur += sizeof(float)*ne10*n_tasks;
                    } break;
                default:
                    break;
            }
        }

        work_size = MAX(work_size, cur);
    }

    if (work_size > 0) {
        work_size += CACHE_LINE_SIZE*(n_threads);
    }

    cplan.threadpool = threadpool;
    cplan.n_threads  = MIN(max_tasks, n_threads);
    cplan.work_size  = work_size;
    cplan.work_data  = NULL;

    return cplan;
}


// ---------------------------------------------------------------------------------------------
// ggml-cpu backend-local fusion passes  (INF-70 SYNC-12)
//
// A fusion pass observes the ALREADY-BUILT graph from inside the CPU node loop and, when its
// pattern matches, performs the work of several graph nodes itself.  It NEVER mutates the graph
// and adds nothing to the ggml op/tensor ABI.
//
// Backend safety is structural, not defensive: a pass exists only inside ggml-cpu.c, so a graph
// carrying a fusable pattern is byte-identical to one that does not, and CUDA / Metal / SYCL /
// Vulkan simply execute the unfused nodes.  There is no flag for another backend to ignore and
// therefore no silent-wrong-answer path.  (Contrast: encoding the fusion in the graph -- e.g. an
// extra src slot only the CPU kernel reads -- makes every backend that ignores it silently skip
// the write-back.  Do not do that.)  This mirrors ggml_cuda_try_fuse() in ggml-cuda.cu.
//
// INVARIANTS a pass must satisfy:
//  1. PURE MATCH.  The matcher must be a deterministic function of the graph alone.  Every one of
//     the nth threads runs it independently and they must all reach the same decision, or the
//     graph barriers at the bottom of the node loop stop being matched and the threadpool
//     deadlocks or races.  No RNG, no time, no per-thread state, no writes from the matcher.
//  2. NO GRAPH WRITES.  Matchers take a const cgraph.
//  3. HANDLED means the pass produced every byte the elided nodes would have produced.
//  4. skip_barrier may be set ONLY when the pass wrote nothing at all (a pure elision).  It drops
//     the trailing graph barrier for that node, which is sound precisely because no data
//     dependency crosses a node that touched no memory.
//  5. Anything not strictly a no-op is gated by its own env knob, default OFF, resolved once in
//     ggml_cpu_init().
// ---------------------------------------------------------------------------------------------

static bool ggml_cpu_disable_fusion = false;  // initialized once in ggml_cpu_init(), read-only afterwards

// pass enables, resolved once in ggml_cpu_init(), read-only afterwards
static bool ggml_cpu_fuse_gdn_state = false;  // GGML_CPU_FUSE_GDN_STATE
static bool ggml_cpu_fuse_empty     = false;  // GGML_CPU_FUSE_EMPTY
static bool ggml_cpu_fuse_debug     = false;  // GGML_CPU_FUSE_DEBUG -- dump per-pass hit counts at exit

#ifdef GGML_CPU_PROF
// per-op wall-time profiling, profiling build only; enabled via GGML_CPU_PROF=1
//
// INF-70 D0-b/c + B1 extension (2026-09-02):
//   The original profiler timed ONLY thread-0's compute call, which sits BEFORE the graph
//   barrier at the bottom of the node loop -- so it could not see barrier wait or straggler
//   imbalance.  We now take three thread-0 timestamps per node: before compute, after compute,
//   and AFTER the graph barrier.  compute = t1-t0, wall = t2-t0, (wall-compute) = barrier +
//   imbalance.  Sums are kept per op type, per node index (decode graphs are shape-stable, so
//   node index is a stable identity across tokens) and per weight path (dense mul_mat /
//   expert mul_mat_id / lm_head).  Everything is accumulated across graph evaluations and
//   dumped once at exit, so prefill can be excluded with GGML_CPU_PROF_SKIP.
//
// Env:
//   GGML_CPU_PROF            enable (any value)
//   GGML_CPU_PROF_SKIP=N     do not accumulate the first N graph evaluations (prefill/warmup)
//   GGML_CPU_PROF_NODES      dump the node table (ALL nodes, no 260 cap)
//   GGML_CPU_PROF_NODES_FILE path for the node table (default stderr)
//   GGML_CPU_PROF_MM         per-call [mm_prof]/[mmid_prof] in-function lines (very noisy)
static uint64_t ggml_cpu_prof_ns[GGML_OP_COUNT];
static uint64_t ggml_cpu_prof_cnt[GGML_OP_COUNT];
static uint64_t ggml_cpu_prof_wall_ns[GGML_OP_COUNT];
static uint64_t ggml_cpu_prof_t1_cnt[GGML_OP_COUNT];   // nodes of this op with n_tasks == 1
static uint64_t ggml_cpu_prof_fused_ns = 0;
static uint64_t ggml_cpu_prof_fused_wall_ns = 0;
static uint64_t ggml_cpu_prof_fused_cnt = 0;
static uint64_t ggml_cpu_prof_total_ns = 0;
static uint64_t ggml_cpu_prof_total_wall_ns = 0;
static int      ggml_cpu_prof_enabled = -1;

// per-node-index accumulators (stable across shape-identical decode graphs)
static uint64_t * ggml_cpu_prof_node_ns      = NULL;
static uint64_t * ggml_cpu_prof_node_wall_ns = NULL;
static uint64_t * ggml_cpu_prof_node_cnt     = NULL;
static int        ggml_cpu_prof_node_cap     = 0;

// weight-path totals: 0 = dense mul_mat, 1 = expert mul_mat_id, 2 = lm_head mul_mat
#define GGML_CPU_PROF_NPATH 3
static uint64_t ggml_cpu_prof_path_ns[GGML_CPU_PROF_NPATH];
static uint64_t ggml_cpu_prof_path_wall_ns[GGML_CPU_PROF_NPATH];
static uint64_t ggml_cpu_prof_path_bytes[GGML_CPU_PROF_NPATH];
static uint64_t ggml_cpu_prof_path_cnt[GGML_CPU_PROF_NPATH];

// INF-70 C2: the atexit dump must NEVER dereference the cgraph -- by then llama_free has
// released the sched's context and the tensors are freed memory (this is the campaign's
// "post-compute dump of freed memory" failure class, and it did segfault the first run).
// Snapshot everything the dump needs, once, at setup time.
struct ggml_cpu_prof_meta {
    int     op;
    int     s0_type;
    int64_t s0_ne[3];
    int64_t ne[3];
    char    name[GGML_MAX_NAME];
};
static struct ggml_cpu_prof_meta * ggml_cpu_prof_meta = NULL;
static int                         ggml_cpu_prof_meta_n = 0;

static uint64_t ggml_cpu_prof_graph_idx     = 0;   // graph evaluations seen
static uint64_t ggml_cpu_prof_graphs_acc    = 0;   // graph evaluations accumulated
static int      ggml_cpu_prof_skip          = -1;
static int      ggml_cpu_prof_nodes_written = 0;
static int      ggml_cpu_prof_atexit_done   = 0;
static int ggml_cpu_prof_last_nnodes = 0;

// lm_head identification: the MUL_MAT whose src0 has ne[1] == the vocab size.  We do not
// hardcode 248320: the widest MUL_MAT src0 ne[1] in the graph is the output projection.
static int64_t ggml_cpu_prof_lm_head_ne1 = 0;

static bool ggml_cpu_prof_is_enabled(void) {
    if (ggml_cpu_prof_enabled < 0) {
        ggml_cpu_prof_enabled = getenv("GGML_CPU_PROF") != NULL ? 1 : 0;
    }
    return ggml_cpu_prof_enabled != 0;
}

static void ggml_cpu_prof_reset(void) __attribute__((unused));
static void ggml_cpu_prof_reset(void) {
    memset(ggml_cpu_prof_ns, 0, sizeof(ggml_cpu_prof_ns));
    memset(ggml_cpu_prof_cnt, 0, sizeof(ggml_cpu_prof_cnt));
    memset(ggml_cpu_prof_wall_ns, 0, sizeof(ggml_cpu_prof_wall_ns));
    memset(ggml_cpu_prof_t1_cnt, 0, sizeof(ggml_cpu_prof_t1_cnt));
    memset(ggml_cpu_prof_path_ns, 0, sizeof(ggml_cpu_prof_path_ns));
    memset(ggml_cpu_prof_path_wall_ns, 0, sizeof(ggml_cpu_prof_path_wall_ns));
    memset(ggml_cpu_prof_path_bytes, 0, sizeof(ggml_cpu_prof_path_bytes));
    memset(ggml_cpu_prof_path_cnt, 0, sizeof(ggml_cpu_prof_path_cnt));
    if (ggml_cpu_prof_node_cap > 0) {
        memset(ggml_cpu_prof_node_ns,      0, ggml_cpu_prof_node_cap*sizeof(uint64_t));
        memset(ggml_cpu_prof_node_wall_ns, 0, ggml_cpu_prof_node_cap*sizeof(uint64_t));
        memset(ggml_cpu_prof_node_cnt,     0, ggml_cpu_prof_node_cap*sizeof(uint64_t));
    }
    ggml_cpu_prof_fused_ns = 0;
    ggml_cpu_prof_fused_wall_ns = 0;
    ggml_cpu_prof_fused_cnt = 0;
    ggml_cpu_prof_total_ns = 0;
    ggml_cpu_prof_total_wall_ns = 0;
    ggml_cpu_prof_graphs_acc = 0;
}

static int ggml_cpu_prof_skip_graphs(void) {
    if (ggml_cpu_prof_skip < 0) {
        const char * e = getenv("GGML_CPU_PROF_SKIP");
        ggml_cpu_prof_skip = e ? atoi(e) : 0;
    }
    return ggml_cpu_prof_skip;
}

static void ggml_cpu_prof_ensure_nodes(int n) {
    if (n <= ggml_cpu_prof_node_cap) {
        return;
    }
    const int cap = n;
    ggml_cpu_prof_node_ns      = (uint64_t *) realloc(ggml_cpu_prof_node_ns,      cap*sizeof(uint64_t));
    ggml_cpu_prof_node_wall_ns = (uint64_t *) realloc(ggml_cpu_prof_node_wall_ns, cap*sizeof(uint64_t));
    ggml_cpu_prof_node_cnt     = (uint64_t *) realloc(ggml_cpu_prof_node_cnt,     cap*sizeof(uint64_t));
    memset(ggml_cpu_prof_node_ns      + ggml_cpu_prof_node_cap, 0, (cap - ggml_cpu_prof_node_cap)*sizeof(uint64_t));
    memset(ggml_cpu_prof_node_wall_ns + ggml_cpu_prof_node_cap, 0, (cap - ggml_cpu_prof_node_cap)*sizeof(uint64_t));
    memset(ggml_cpu_prof_node_cnt     + ggml_cpu_prof_node_cap, 0, (cap - ggml_cpu_prof_node_cap)*sizeof(uint64_t));
    ggml_cpu_prof_node_cap = cap;
}

static void ggml_cpu_prof_snapshot_meta(const struct ggml_cgraph * cgraph) {
    free(ggml_cpu_prof_meta);
    ggml_cpu_prof_meta   = (struct ggml_cpu_prof_meta *) calloc(cgraph->n_nodes, sizeof(struct ggml_cpu_prof_meta));
    ggml_cpu_prof_meta_n = ggml_cpu_prof_meta ? cgraph->n_nodes : 0;
    for (int i = 0; i < ggml_cpu_prof_meta_n; i++) {
        const struct ggml_tensor * nd = cgraph->nodes[i];
        const struct ggml_tensor * s0 = nd->src[0];
        struct ggml_cpu_prof_meta * m = &ggml_cpu_prof_meta[i];
        m->op      = (int) nd->op;
        m->s0_type = s0 ? (int) s0->type : -1;
        for (int k = 0; k < 3; k++) {
            m->s0_ne[k] = s0 ? s0->ne[k] : 0;
            m->ne[k]    = nd->ne[k];
        }
        snprintf(m->name, sizeof(m->name), "%s", nd->name);
    }
}

// bytes of weight streamed by one node, 0 for ops that do not stream a weight slab
static uint64_t ggml_cpu_prof_node_bytes(const struct ggml_tensor * node) {
    if (node->op == GGML_OP_MUL_MAT) {
        return (uint64_t) ggml_nbytes(node->src[0]);
    }
    if (node->op == GGML_OP_MUL_MAT_ID) {
        const struct ggml_tensor * ids = node->src[2];
        if (!ids) {
            return 0;
        }
        // only the used experts' slabs are touched: n_expert_used * n_tokens * nb02
        return (uint64_t) ids->ne[0] * (uint64_t) ids->ne[1] * (uint64_t) node->src[0]->nb[2];
    }
    return 0;
}

// path index for a node, or -1
static int ggml_cpu_prof_node_path(const struct ggml_tensor * node) {
    if (node->op == GGML_OP_MUL_MAT_ID) {
        return 1;
    }
    if (node->op == GGML_OP_MUL_MAT) {
        if (ggml_cpu_prof_lm_head_ne1 != 0 && node->src[0]->ne[1] == ggml_cpu_prof_lm_head_ne1) {
            return 2;
        }
        return 0;
    }
    return -1;
}

static void ggml_cpu_prof_write_nodes(const struct ggml_cgraph * cgraph, int n_threads) {
    if (getenv("GGML_CPU_PROF_NODES") == NULL || ggml_cpu_prof_nodes_written) {
        return;
    }
    ggml_cpu_prof_nodes_written = 1;

    const char * path = getenv("GGML_CPU_PROF_NODES_FILE");
    FILE * f = stderr;
    if (path) {
        f = fopen(path, "w");
        if (!f) { f = stderr; }
    }

    fprintf(f, "# INF-70 node table: one line per graph node, n_nodes=%d n_threads=%d\n", cgraph->n_nodes, n_threads);
    fprintf(f, "# idx\top\tempty\tcompute\tn_tasks\tdst_type\tdst_ne\t"
               "src0_op\tsrc0_type\tsrc0_ne\tsrc0_view\tsrc0_cont\t"
               "src1_op\tsrc1_type\tsrc1_ne\tsrc1_view\tsrc1_cont\t"
               "src2_ne0\tsrc2_ne1\tbytes\tname\n");
    for (int i = 0; i < cgraph->n_nodes; i++) {
        const struct ggml_tensor * n = cgraph->nodes[i];
        const struct ggml_tensor * s0 = n->src[0];
        const struct ggml_tensor * s1 = n->src[1];
        const struct ggml_tensor * s2 = n->src[2];
        const int empty   = ggml_op_is_empty(n->op) ? 1 : 0;
        const int compute = (n->flags & GGML_TENSOR_FLAG_COMPUTE) ? 1 : 0;
        fprintf(f,
            "%d\t%s\t%d\t%d\t%d\t%s\t%lld,%lld,%lld,%lld\t"
            "%s\t%s\t%lld,%lld,%lld,%lld\t%d\t%d\t"
            "%s\t%s\t%lld,%lld,%lld,%lld\t%d\t%d\t"
            "%lld\t%lld\t%llu\t%s\n",
            i, ggml_op_name(n->op), empty, compute,
            ggml_get_n_tasks((struct ggml_tensor *) n, n_threads),
            ggml_type_name(n->type),
            (long long) n->ne[0], (long long) n->ne[1], (long long) n->ne[2], (long long) n->ne[3],
            s0 ? ggml_op_name(s0->op) : "-", s0 ? ggml_type_name(s0->type) : "-",
            s0 ? (long long) s0->ne[0] : 0, s0 ? (long long) s0->ne[1] : 0,
            s0 ? (long long) s0->ne[2] : 0, s0 ? (long long) s0->ne[3] : 0,
            s0 ? (s0->view_src != NULL) : 0, s0 ? ggml_is_contiguous(s0) : 0,
            s1 ? ggml_op_name(s1->op) : "-", s1 ? ggml_type_name(s1->type) : "-",
            s1 ? (long long) s1->ne[0] : 0, s1 ? (long long) s1->ne[1] : 0,
            s1 ? (long long) s1->ne[2] : 0, s1 ? (long long) s1->ne[3] : 0,
            s1 ? (s1->view_src != NULL) : 0, s1 ? ggml_is_contiguous(s1) : 0,
            s2 ? (long long) s2->ne[0] : 0, s2 ? (long long) s2->ne[1] : 0,
            (unsigned long long) ggml_cpu_prof_node_bytes(n),
            n->name);
    }
    fflush(f);
    if (f != stderr) {
        fclose(f);
    }
}

static void ggml_cpu_prof_dump(void) {
    const uint64_t G = ggml_cpu_prof_graphs_acc;
    if (G == 0 || ggml_cpu_prof_total_wall_ns == 0) {
        return;
    }
    const double g = (double) G;

    fprintf(stderr, "\n[cpu_prof] ==== INF-70 profile: %llu graph evals accumulated (skipped first %d) ====\n",
            (unsigned long long) G, ggml_cpu_prof_skip_graphs());
    fprintf(stderr, "[cpu_prof] per graph eval: thread0_compute %.3f ms | wall(compute+barrier) %.3f ms | n_nodes %d\n",
            ggml_cpu_prof_total_ns/1e3/g, ggml_cpu_prof_total_wall_ns/1e3/g,
            ggml_cpu_prof_last_nnodes);
    fprintf(stderr, "[cpu_prof] fused (RMS_NORM+MUL): %.1f ops/eval  compute %.3f ms  wall %.3f ms\n",
            ggml_cpu_prof_fused_cnt/g, ggml_cpu_prof_fused_ns/1e3/g, ggml_cpu_prof_fused_wall_ns/1e3/g);
    fprintf(stderr, "[cpu_prof] SYNC measured ggml_barrier() calls per graph eval: %.1f\n",
            ggml_cpu_prof_barriers/g);

    // ---- per op type, sorted by wall ----
    struct { int op; uint64_t ns; uint64_t wall; uint64_t cnt; uint64_t t1; } rows[GGML_OP_COUNT];
    int n_rows = 0;
    for (int i = 0; i < GGML_OP_COUNT; i++) {
        if (ggml_cpu_prof_cnt[i] > 0) {
            rows[n_rows].op   = i;
            rows[n_rows].ns   = ggml_cpu_prof_ns[i];
            rows[n_rows].wall = ggml_cpu_prof_wall_ns[i];
            rows[n_rows].cnt  = ggml_cpu_prof_cnt[i];
            rows[n_rows].t1   = ggml_cpu_prof_t1_cnt[i];
            n_rows++;
        }
    }
    for (int i = 0; i < n_rows; i++) {
        for (int j = i + 1; j < n_rows; j++) {
            if (rows[j].wall > rows[i].wall) {
                typeof(rows[0]) tmp = rows[i]; rows[i] = rows[j]; rows[j] = tmp;
            }
        }
    }
    fprintf(stderr, "[cpu_prof] OPTABLE\top\tcount_per_eval\tcompute_ms\twall_ms\tdelta_ms\tpct_wall\tn_tasks1_count\tavg_wall_us\n");
    for (int i = 0; i < n_rows; i++) {
        fprintf(stderr, "[cpu_prof] OPROW\t%s\t%.1f\t%.4f\t%.4f\t%.4f\t%.2f\t%.1f\t%.2f\n",
                ggml_op_name(rows[i].op), rows[i].cnt/g,
                rows[i].ns/1e3/g,   /* ggml_time_us() -> counters are microseconds */
                rows[i].wall/1e3/g,
                (double)(rows[i].wall - rows[i].ns)/1e3/g,
                100.0*rows[i].wall/ggml_cpu_prof_total_wall_ns,
                rows[i].t1/g,
                (double) rows[i].wall / rows[i].cnt);
    }

    // ---- per weight path ----
    static const char * pname[GGML_CPU_PROF_NPATH] = { "dense_mul_mat", "expert_mul_mat_id", "lm_head" };
    fprintf(stderr, "[cpu_prof] PATHTABLE\tpath\tcalls_per_eval\tcompute_ms\twall_ms\tbytes_per_eval\tGBs_on_compute\tGBs_on_wall\n");
    for (int i = 0; i < GGML_CPU_PROF_NPATH; i++) {
        if (ggml_cpu_prof_path_cnt[i] == 0) continue;
        const double bytes = ggml_cpu_prof_path_bytes[i]/g;
        const double cms   = ggml_cpu_prof_path_ns[i]/1e3/g;
        const double wms   = ggml_cpu_prof_path_wall_ns[i]/1e3/g;
        fprintf(stderr, "[cpu_prof] PATHROW\t%s\t%.1f\t%.4f\t%.4f\t%.0f\t%.2f\t%.2f\n",
                pname[i], ggml_cpu_prof_path_cnt[i]/g, cms, wms, bytes,
                cms > 0 ? bytes/1e9/(cms/1e3) : 0.0,
                wms > 0 ? bytes/1e9/(wms/1e3) : 0.0);
    }

    // ---- per node index, sorted by wall, top 64 (from the SNAPSHOT, never the cgraph) ----
    if (ggml_cpu_prof_meta_n > 0 && ggml_cpu_prof_node_cap > 0) {
        int n = ggml_cpu_prof_meta_n < ggml_cpu_prof_node_cap ? ggml_cpu_prof_meta_n : ggml_cpu_prof_node_cap;
        int * idx = (int *) malloc(n*sizeof(int));
        int m = 0;
        for (int i = 0; i < n; i++) {
            if (ggml_cpu_prof_node_cnt[i] > 0) idx[m++] = i;
        }
        for (int i = 0; i < m; i++) {
            for (int j = i + 1; j < m; j++) {
                if (ggml_cpu_prof_node_wall_ns[idx[j]] > ggml_cpu_prof_node_wall_ns[idx[i]]) {
                    int t = idx[i]; idx[i] = idx[j]; idx[j] = t;
                }
            }
        }
        fprintf(stderr, "[cpu_prof] NODETABLE\tidx\top\tname\tsrc0_type\tsrc0_ne\tdst_ne\tcompute_us\twall_us\tdelta_us\n");
        for (int i = 0; i < m && i < 64; i++) {
            const int k = idx[i];
            const struct ggml_cpu_prof_meta * md = &ggml_cpu_prof_meta[k];
            fprintf(stderr, "[cpu_prof] NODEROW\t%d\t%s\t%s\t%s\t%lld,%lld,%lld\t%lld,%lld,%lld\t%.2f\t%.2f\t%.2f\n",
                    k, ggml_op_name((enum ggml_op) md->op), md->name,
                    md->s0_type >= 0 ? ggml_type_name((enum ggml_type) md->s0_type) : "-",
                    (long long) md->s0_ne[0], (long long) md->s0_ne[1], (long long) md->s0_ne[2],
                    (long long) md->ne[0], (long long) md->ne[1], (long long) md->ne[2],
                    ggml_cpu_prof_node_ns[k]/g, ggml_cpu_prof_node_wall_ns[k]/g,
                    (double)(ggml_cpu_prof_node_wall_ns[k] - ggml_cpu_prof_node_ns[k])/g);
        }
        const char * pf = getenv("GGML_CPU_PROF_PERNODE_FILE");
        if (pf) {
            FILE * f = fopen(pf, "w");
            if (f) {
                fprintf(f, "idx\top\tname\tsrc0_type\tsrc0_ne0\tsrc0_ne1\tsrc0_ne2\tdst_ne0\tdst_ne1\tdst_ne2\tcompute_us\twall_us\tevals\n");
                for (int i = 0; i < n; i++) {
                    if (ggml_cpu_prof_node_cnt[i] == 0) continue;
                    const struct ggml_cpu_prof_meta * md = &ggml_cpu_prof_meta[i];
                    fprintf(f, "%d\t%s\t%s\t%s\t%lld\t%lld\t%lld\t%lld\t%lld\t%lld\t%.3f\t%.3f\t%llu\n",
                            i, ggml_op_name((enum ggml_op) md->op), md->name,
                            md->s0_type >= 0 ? ggml_type_name((enum ggml_type) md->s0_type) : "-",
                            (long long) md->s0_ne[0], (long long) md->s0_ne[1], (long long) md->s0_ne[2],
                            (long long) md->ne[0], (long long) md->ne[1], (long long) md->ne[2],
                            ggml_cpu_prof_node_ns[i]/g, ggml_cpu_prof_node_wall_ns[i]/g,
                            (unsigned long long) ggml_cpu_prof_node_cnt[i]);
                }
                fclose(f);
            }
        }
        free(idx);
    }
    fflush(stderr);
}

static void ggml_cpu_prof_atexit(void) {
    ggml_cpu_prof_dump();
}
#endif

// O(1) use count for an arbitrary tensor in the graph (ggml_node_get_use_count only takes a
// node index; the tensors we must prove single-use here are views, which are not nodes).
// Returns -1 when the tensor is not in the graph's hash set.
static int32_t ggml_cpu_tensor_use_count(const struct ggml_cgraph * cgraph, const struct ggml_tensor * t) {
    const size_t hash_pos = ggml_hash_find(&cgraph->visited_hash_set, t);
    if (hash_pos == GGML_HASHSET_FULL || !ggml_bitset_get(cgraph->visited_hash_set.used, hash_pos)) {
        return -1;
    }
    return cgraph->use_counts[hash_pos];
}

// index of the first node after node_n that actually computes something, or -1
static int ggml_cpu_next_real_node(const struct ggml_cgraph * cgraph, int node_n) {
    for (int j = node_n + 1; j < cgraph->n_nodes; ++j) {
        const struct ggml_tensor * n = cgraph->nodes[j];
        if (ggml_op_is_empty(n->op) || (n->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            continue;
        }
        return j;
    }
    return -1;
}

// -------------------------------------------------------------------------------------------
// pass: rms_norm_mul   (always on; pre-existing behaviour)
// -------------------------------------------------------------------------------------------
static bool ggml_cpu_fuse_pass_rms_norm_mul(
        const struct ggml_cgraph * cgraph, int node_n,
        const struct ggml_compute_params * params,
        int * n_extra, bool * skip_barrier) {
    GGML_UNUSED(skip_barrier);

    struct ggml_tensor * node = cgraph->nodes[node_n];

    const enum ggml_op fuse_ops[] = { GGML_OP_RMS_NORM, GGML_OP_MUL };
    if (!ggml_can_fuse(cgraph, node_n, fuse_ops, 2)) {
        return false;
    }

    struct ggml_tensor * mul_node = cgraph->nodes[node_n + 1];
    const struct ggml_tensor * mul_w = (mul_node->src[0] == node) ? mul_node->src[1] : mul_node->src[0];

    if (node->src[0]->type != GGML_TYPE_F32 || mul_node->type != GGML_TYPE_F32 ||
        mul_w->type != GGML_TYPE_F32 || mul_w->ne[0] != node->ne[0] || mul_w->nb[0] != sizeof(float)) {
        return false;
    }

    ggml_compute_forward_rms_norm_mul_fused(params, node, mul_node);
    *n_extra = 1;
    return true;
}

// -------------------------------------------------------------------------------------------
// pass: gdn_state_cache   (GGML_CPU_FUSE_GDN_STATE=1, default OFF)
//
//   GATED_DELTA_NET(K==1) -> [views/no-ops] -> CPY( view(gdn, tail_off) -> cache_view )
//
// The op already computes the final recurrent state; it merely writes it into its own dst tail,
// from where the graph copies it into the recurrent-state cache slot.  The pass hands the cache
// slot to the kernel, which writes it in place, and the CPY node is not executed.
//
// This is the K==1 counterpart of ggml_cuda_try_gdn_cache_fusion(), which bails at K<=1 -- see
// tmp/inf70/agents/sync12/REPORT.md: the CUDA kernel's !keep_rs epilogue is already the fused
// form, so that guard is scope, not soundness.
// -------------------------------------------------------------------------------------------
static bool ggml_cpu_fuse_pass_gdn_state_cache(
        const struct ggml_cgraph * cgraph, int node_n,
        const struct ggml_compute_params * params,
        int * n_extra, bool * skip_barrier) {
    GGML_UNUSED(skip_barrier);

    struct ggml_tensor * gdn = cgraph->nodes[node_n];

    // the kernel skips the snapshot tail, so the gdn output must not be a graph output
    if (gdn->type != GGML_TYPE_F32 || (gdn->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        return false;
    }
    if (gdn->src[0] == NULL || gdn->src[0]->type != GGML_TYPE_F32) {
        return false;
    }

    // this pass implements only the K==1 (no rollback snapshots) shape
    if (ggml_get_op_params_i32(gdn, 0) != 1) {
        return false;
    }

    const struct ggml_tensor * src_v    = gdn->src[2];
    const struct ggml_tensor * src_st   = gdn->src[5];
    if (src_v == NULL || src_st == NULL) {
        return false;
    }
    const int64_t S_v      = src_v->ne[0];
    const int64_t H        = src_v->ne[1];
    const int64_t n_tokens = src_v->ne[2];
    const int64_t n_seqs   = src_v->ne[3];
    const int64_t D        = S_v * S_v * H;

    // the state read source must be a distinct buffer from the cache we are about to write:
    // the kernel reads s_in for (head, seq) then writes s_out for the same (head, seq).  The
    // graph feeds src[5] from a GET_ROWS gather, never from the cache tensor itself; require it.
    if (src_st->op == GGML_OP_NONE && src_st->view_src == NULL) {
        return false;
    }

    // snapshot tail starts right after the attention scores
    const size_t tail_off = ggml_row_size(GGML_TYPE_F32, S_v * H * n_tokens * n_seqs);

    const int j = ggml_cpu_next_real_node(cgraph, node_n);
    if (j < 0) {
        return false;
    }
    const struct ggml_tensor * cpy = cgraph->nodes[j];
    if (cpy->op != GGML_OP_CPY || (cpy->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        return false;
    }

    const struct ggml_tensor * src = cpy->src[0]; // view of the gdn state tail
    const struct ggml_tensor * dst = cpy->src[1]; // cache view the kernel will write to
    if (src == NULL || dst == NULL) {
        return false;
    }

    // src must be exactly this gdn's state tail, and nothing else may read it
    if (src->op != GGML_OP_VIEW || src->view_src != gdn || src->view_offs != tail_off ||
        src->type != GGML_TYPE_F32 || !ggml_is_contiguous(src) ||
        ggml_nelements(src) != D * n_seqs ||
        ggml_cpu_tensor_use_count(cgraph, src) != 1) {
        return false;
    }

    // dst is the [D, n_seqs] cache view; nb[1] is the per-seq stride the kernel will use
    if (dst->type != GGML_TYPE_F32 || dst->data == NULL || cpy->type != GGML_TYPE_F32 ||
        dst->ne[0] != D || dst->ne[1] != n_seqs || dst->ne[2] != 1 || dst->ne[3] != 1 ||
        dst->nb[0] != ggml_type_size(GGML_TYPE_F32) ||
        dst->nb[1] < (size_t) ggml_row_size(GGML_TYPE_F32, D) ||
        dst->nb[1] % sizeof(float) != 0) {
        return false;
    }

    ggml_compute_forward_gated_delta_net_fused_cache(
            params, gdn, (float *) dst->data, (int64_t) (dst->nb[1] / sizeof(float)));

    *n_extra = j - node_n;
    return true;
}

// -------------------------------------------------------------------------------------------
// pass: empty   (GGML_CPU_FUSE_EMPTY=1, default OFF)
//
// A node whose dst has zero elements produces nothing.  Its ggml_compute_forward is already a
// no-op for every such op, but the node still costs a full nth-way graph barrier.  Elide the
// node AND its barrier -- sound because no data dependency can cross a node that wrote nothing,
// and safe because every thread reaches the identical decision (invariant 1).
//
// This is the second consumer of the entry point, and the one that shows it generalises past
// "two ops become one": a pass may also delete work outright.
// -------------------------------------------------------------------------------------------
static bool ggml_cpu_fuse_pass_empty(
        const struct ggml_cgraph * cgraph, int node_n,
        const struct ggml_compute_params * params,
        int * n_extra, bool * skip_barrier) {
    GGML_UNUSED(params);

    const struct ggml_tensor * node = cgraph->nodes[node_n];

    if (ggml_nelements(node) != 0 || (node->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        return false;
    }
    *n_extra      = 0;
    *skip_barrier = true;
    return true;
}

// -------------------------------------------------------------------------------------------
// the registry
// -------------------------------------------------------------------------------------------
struct ggml_cpu_fusion_pass {
    const char * name;
    // op of cgraph->nodes[node_n] that arms this pass; GGML_OP_COUNT = any op
    enum ggml_op trigger;
    // NULL = always enabled, else a pointer to the pass's resolved enable flag
    const bool * enabled;
    bool (*run)(const struct ggml_cgraph * cgraph, int node_n,
                const struct ggml_compute_params * params,
                int * n_extra, bool * skip_barrier);
};

static const struct ggml_cpu_fusion_pass ggml_cpu_fusion_passes[] = {
    { "rms_norm_mul",   GGML_OP_RMS_NORM,        NULL,                     ggml_cpu_fuse_pass_rms_norm_mul   },
    { "gdn_state_cache",GGML_OP_GATED_DELTA_NET, &ggml_cpu_fuse_gdn_state, ggml_cpu_fuse_pass_gdn_state_cache},
    { "empty",          GGML_OP_COUNT,           &ggml_cpu_fuse_empty,     ggml_cpu_fuse_pass_empty          },
};

#define GGML_CPU_N_FUSION_PASSES (sizeof(ggml_cpu_fusion_passes)/sizeof(ggml_cpu_fusion_passes[0]))

// hit counters, incremented by thread 0 only (so no atomics); diagnostic, GGML_CPU_FUSE_DEBUG=1
static uint64_t ggml_cpu_fusion_hits[GGML_CPU_N_FUSION_PASSES] = { 0 };
static bool     ggml_cpu_fusion_atexit_done = false;

static void ggml_cpu_fusion_dump(void) {
    for (size_t p = 0; p < GGML_CPU_N_FUSION_PASSES; ++p) {
        fprintf(stderr, "[cpu_fuse] pass=%-16s enabled=%d hits_thread0=%llu\n",
                ggml_cpu_fusion_passes[p].name,
                ggml_cpu_fusion_passes[p].enabled == NULL ? 1 : (int) *ggml_cpu_fusion_passes[p].enabled,
                (unsigned long long) ggml_cpu_fusion_hits[p]);
    }
    fflush(stderr);
}

// Try the registered fusion passes against cgraph->nodes[node_n].
// Returns 1 when a pass handled the node (do NOT call ggml_compute_forward), else 0.
//   *n_extra      = additional graph nodes consumed by the pass (0 for a pure elision)
//   *skip_barrier = the pass wrote nothing; the trailing graph barrier may be dropped
static int ggml_cpu_try_fuse_ops(
        const struct ggml_cgraph * cgraph,
        const int node_n,
        const struct ggml_compute_params * params,
        const struct ggml_cplan * cplan,
        int * n_extra,
        bool * skip_barrier) {

    *n_extra      = 0;
    *skip_barrier = false;

    if (ggml_cpu_disable_fusion || cplan->use_ref) {
        return 0;
    }

    const struct ggml_tensor * node = cgraph->nodes[node_n];

    for (size_t p = 0; p < sizeof(ggml_cpu_fusion_passes)/sizeof(ggml_cpu_fusion_passes[0]); ++p) {
        const struct ggml_cpu_fusion_pass * pass = &ggml_cpu_fusion_passes[p];

        if (pass->enabled != NULL && !*pass->enabled) {
            continue;
        }
        if (pass->trigger != GGML_OP_COUNT && node->op != pass->trigger) {
            continue;
        }
        if (pass->run(cgraph, node_n, params, n_extra, skip_barrier)) {
            if (ggml_cpu_fuse_debug && params->ith == 0) {
                ggml_cpu_fusion_hits[p]++;
                if (!ggml_cpu_fusion_atexit_done) {
                    ggml_cpu_fusion_atexit_done = true;
                    atexit(ggml_cpu_fusion_dump);
                }
            }
            return 1;
        }
        *n_extra      = 0;
        *skip_barrier = false;
    }

    return 0;
}

static thread_ret_t ggml_graph_compute_thread(void * data) {
    struct ggml_compute_state * state = (struct ggml_compute_state *) data;
    struct ggml_threadpool    * tp    = state->threadpool;

    const struct ggml_cgraph * cgraph = tp->cgraph;
    const struct ggml_cplan  * cplan  = tp->cplan;

#ifdef GGML_USE_CPU_RISCV64_SPACEMIT
    ggml_backend_cpu_riscv64_spacemit_set_numa_thread_affinity(state->ith);
#else
    set_numa_thread_affinity(state->ith);
#endif

    struct ggml_compute_params params = {
        /*.ith        =*/ state->ith,
        /*.nth        =*/ atomic_load_explicit(&tp->n_graph, memory_order_relaxed) & GGML_THREADPOOL_N_THREADS_MASK,
        /*.wsize      =*/ cplan->work_size,
        /*.wdata      =*/ cplan->work_data,
        /*.threadpool =*/ tp,
        /*.use_ref    =*/ cplan->use_ref,
    };

#ifdef GGML_USE_OPENMP
    GGML_PRINT_DEBUG("thread #%d compute-start cplan %p\n", state->ith, (const void *)cplan);
#else
    GGML_PRINT_DEBUG("thread #%d compute-start cplan %p last-graph %d\n", state->ith, (const void *)cplan, state->last_graph);
#endif

#ifdef GGML_CPU_PROF
    // INF-70: one-time-per-graph-eval setup, thread 0 only.  prof_acc says whether this
    // evaluation is accumulated (GGML_CPU_PROF_SKIP excludes prefill/warmup graphs).
    int prof_acc = 0;
    if (state->ith == 0 && ggml_cpu_prof_is_enabled()) {
        const uint64_t gi = ggml_cpu_prof_graph_idx++;
        prof_acc = (gi >= (uint64_t) ggml_cpu_prof_skip_graphs()) ? 1 : 0;
        ggml_cpu_prof_barrier_on = prof_acc;
        if (prof_acc) {
            if (ggml_cpu_prof_graphs_acc == 0) {
                // widest MUL_MAT src0 ne[1] in the graph == the output projection (lm_head)
                for (int i = 0; i < cgraph->n_nodes; i++) {
                    const struct ggml_tensor * nd = cgraph->nodes[i];
                    if (nd->op == GGML_OP_MUL_MAT && nd->src[0] &&
                        nd->src[0]->ne[1] > ggml_cpu_prof_lm_head_ne1) {
                        ggml_cpu_prof_lm_head_ne1 = nd->src[0]->ne[1];
                    }
                }
                ggml_cpu_prof_snapshot_meta(cgraph);
                ggml_cpu_prof_write_nodes(cgraph, params.nth);
                if (!ggml_cpu_prof_atexit_done) {
                    ggml_cpu_prof_atexit_done = 1;
                    atexit(ggml_cpu_prof_atexit);
                }
            }
            ggml_cpu_prof_ensure_nodes(cgraph->n_nodes);
            ggml_cpu_prof_graphs_acc++;
            ggml_cpu_prof_last_nnodes = cgraph->n_nodes;
        }
        fprintf(stderr, "[cpu_prof] graph_eval idx=%llu n_nodes=%d nth=%d acc=%d\n",
                (unsigned long long) gi, cgraph->n_nodes, params.nth, prof_acc);
    }
#endif

    for (int node_n = 0; node_n < cgraph->n_nodes && atomic_load_explicit(&tp->abort, memory_order_relaxed) != node_n; node_n++) {
        struct ggml_tensor * node = cgraph->nodes[node_n];

        if (ggml_op_is_empty(node->op)) {
            // skip NOPs
            continue;
        }

        if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            continue;
        }

#ifdef GGML_CPU_PROF
        const int64_t t0 = (state->ith == 0 && ggml_cpu_prof_is_enabled() && prof_acc) ? ggml_time_us() : 0;
#endif

#ifdef GGML_CPU_PROF
        const int prof_node_idx = node_n;
#endif

        // TODO: move fused-op detection into ggml_graph_plan so fusion decisions are made once at planning time
        // Try the backend-local fusion passes, fall back to normal compute
        int  fuse_extra   = 0;
        bool skip_barrier = false;
        const int n_fused = ggml_cpu_try_fuse_ops(cgraph, node_n, &params, cplan, &fuse_extra, &skip_barrier);
        if (n_fused > 0) {
            node_n += fuse_extra;
        } else {
            ggml_compute_forward(&params, node);
        }

#ifdef GGML_CPU_PROF
        const int64_t t1 = (t0 != 0) ? ggml_time_us() : 0;
#endif

        if (state->ith == 0 && cplan->abort_callback &&
                cplan->abort_callback(cplan->abort_callback_data)) {
            atomic_store_explicit(&tp->abort, node_n + 1, memory_order_relaxed);
            tp->ec    = GGML_STATUS_ABORTED;
        }

        // skip_barrier is only ever set by a pass that wrote nothing at all, and every thread
        // reaches the same decision (fusion-pass invariant 1), so the barriers stay matched.
        // Suppressed when an abort_callback is installed: thread 0 stores tp->abort between the
        // compute and the barrier, and the barrier is what lets the other threads observe it
        // before the next loop-condition read.
        const bool drop_barrier = skip_barrier && cplan->abort_callback == NULL;
        if (node_n + 1 < cgraph->n_nodes && !drop_barrier) {
            ggml_barrier(state->threadpool);
        }

#ifdef GGML_CPU_PROF
        // INF-70 D0-c: t2 is taken AFTER the graph barrier, so (t2-t0) is this node's
        // wall cost including straggler wait and the barrier itself.  (t1-t0) is thread-0
        // compute only -- what the pre-2026-09-02 profiler could see.
        if (t0 != 0) {
            const uint64_t dt   = (uint64_t)(t1 - t0);
            const uint64_t wall = (uint64_t)(ggml_time_us() - t0);
            ggml_cpu_prof_total_ns      += dt;
            ggml_cpu_prof_total_wall_ns += wall;
            if (prof_node_idx < ggml_cpu_prof_node_cap) {
                ggml_cpu_prof_node_ns[prof_node_idx]      += dt;
                ggml_cpu_prof_node_wall_ns[prof_node_idx] += wall;
                ggml_cpu_prof_node_cnt[prof_node_idx]     += 1;
            }
            if (n_fused > 0) {
                ggml_cpu_prof_fused_ns      += dt;
                ggml_cpu_prof_fused_wall_ns += wall;
                ggml_cpu_prof_fused_cnt++;
            } else {
                ggml_cpu_prof_ns[node->op]      += dt;
                ggml_cpu_prof_wall_ns[node->op] += wall;
                ggml_cpu_prof_cnt[node->op]++;
                if (ggml_get_n_tasks(node, params.nth) == 1) {
                    ggml_cpu_prof_t1_cnt[node->op]++;
                }
                const int path = ggml_cpu_prof_node_path(node);
                if (path >= 0) {
                    ggml_cpu_prof_path_ns[path]      += dt;
                    ggml_cpu_prof_path_wall_ns[path] += wall;
                    ggml_cpu_prof_path_bytes[path]   += ggml_cpu_prof_node_bytes(node);
                    ggml_cpu_prof_path_cnt[path]++;
                }
            }
        }
#endif
    }


#ifdef GGML_USE_OPENMP
    GGML_PRINT_DEBUG("thread #%d compute-done cplan %p\n", state->ith, (const void *)cplan);
#else
    GGML_PRINT_DEBUG("thread #%d compute-done cplan %p last-graph %d\n", state->ith, (const void *)cplan, state->last_graph);
#endif

    ggml_barrier(state->threadpool);

#ifdef GGML_USE_CPU_RISCV64_SPACEMIT
    ggml_backend_cpu_riscv64_spacemit_clear_numa_thread_affinity_threaded(state->ith);
#endif

    return 0;
}

#ifndef GGML_USE_OPENMP

// check if thread is ready to proceed (exit from polling or sleeping)
// returns true if loops should exit, sets state->pending to indicate new work
static inline bool ggml_graph_compute_thread_ready(struct ggml_compute_state * state) {
    struct ggml_threadpool * threadpool = state->threadpool;

    if (state->pending || threadpool->stop || threadpool->pause) { return true; }

    // check for new graph/work
    int n_graph   = atomic_load_explicit(&threadpool->n_graph, memory_order_relaxed);
    int n_threads = n_graph & GGML_THREADPOOL_N_THREADS_MASK;
    if (n_graph != state->last_graph) {
        state->pending    = (state->ith < n_threads);
        state->last_graph = n_graph;
        return true;
    }

    return false;
}

// sync thread state after polling
static inline void ggml_graph_compute_thread_sync(struct ggml_compute_state * state) {
    // TSAN doesn't support standalone fence yet, we use a dummy read-modify-write instead
    #ifdef GGML_TSAN_ENABLED
    atomic_fetch_add_explicit(&state->threadpool->n_graph, 0, memory_order_seq_cst);
    #else
    atomic_thread_fence(memory_order_seq_cst);
    #endif
    UNUSED(state);
}

static inline bool ggml_graph_compute_poll_for_work(struct ggml_compute_state * state) {
    struct ggml_threadpool * threadpool = state->threadpool;

    // This seems to make 0 ... 100 a decent range for polling level across modern processors.
    // Perhaps, we can adjust it dynamically based on load and things.
    const uint64_t n_rounds = 1024UL * 128 * threadpool->poll;

    for (uint64_t i=0; !ggml_graph_compute_thread_ready(state) && i < n_rounds; i++) {
        // No new work. Keep polling.
        ggml_thread_cpu_relax();
    }

    return state->pending;
}

static inline bool ggml_graph_compute_check_for_work(struct ggml_compute_state * state) {
    struct ggml_threadpool * threadpool = state->threadpool;

    if (ggml_graph_compute_poll_for_work(state)) {
        ggml_graph_compute_thread_sync(state);
        return state->pending;
    }

    ggml_mutex_lock_shared(&threadpool->mutex);
    while (!ggml_graph_compute_thread_ready(state)) {
        // No new work. Wait for the signal.
        GGML_PRINT_DEBUG("thread #%d waiting for work (sleeping)\n", state->ith);
        ggml_cond_wait(&threadpool->cond, &threadpool->mutex);
    }
    ggml_mutex_unlock_shared(&threadpool->mutex);

    return state->pending;
}

static thread_ret_t ggml_graph_compute_secondary_thread(void* data) {
    struct ggml_compute_state * state = (struct ggml_compute_state *) data;
    struct ggml_threadpool * threadpool = state->threadpool;

    ggml_thread_apply_priority(threadpool->prio);
    if (ggml_thread_cpumask_is_valid(state->cpumask)) {
        ggml_thread_apply_affinity(state->cpumask);
    }

    while (true) {
        // Check if we need to sleep
        while (threadpool->pause) {
            GGML_PRINT_DEBUG("thread #%d inside pause loop\n", state->ith);
            ggml_mutex_lock_shared(&threadpool->mutex);
            if (threadpool->pause) {
                ggml_cond_wait(&threadpool->cond, &threadpool->mutex);
            }
            GGML_PRINT_DEBUG("thread #%d resuming after wait\n", state->ith);
            ggml_mutex_unlock_shared(&threadpool->mutex);
        }

        // This needs to be checked for after the cond_wait
        if (threadpool->stop) break;

        // Check if there is new work
        // The main thread is the only one that can dispatch new work

        ggml_graph_compute_check_for_work(state);
        if (state->pending) {
            state->pending = false;
            ggml_graph_compute_thread(state);
        }
    }

    return (thread_ret_t) 0;
}

// Start processing new graph
static void ggml_graph_compute_kickoff(struct ggml_threadpool * threadpool, int n_threads)
{
    // Always take the mutex here because the worker threads are doing hybrid poll/wait

    ggml_mutex_lock(&threadpool->mutex);

    // Update the number of active threads and the graph count
    int n_graph = atomic_load_explicit(&threadpool->n_graph, memory_order_relaxed) >> GGML_THREADPOOL_N_THREADS_BITS;
    n_graph = ((n_graph + 1) << GGML_THREADPOOL_N_THREADS_BITS) | (n_threads & GGML_THREADPOOL_N_THREADS_MASK);

    GGML_PRINT_DEBUG("compute-kickoff: n_threads %d n_graph %d\n", n_threads, n_graph);

    // Indicate the graph is ready to be processed
    // We need the full seq-cst fence here because of the polling threads (used in thread_sync)
    atomic_store_explicit(&threadpool->n_graph, n_graph, memory_order_seq_cst);

    if (threadpool->pause) {
       // Update main thread prio and affinity to match the threadpool settings
       ggml_thread_apply_priority(threadpool->prio);
       if (ggml_thread_cpumask_is_valid(threadpool->workers[0].cpumask)) {
           ggml_thread_apply_affinity(threadpool->workers[0].cpumask);
       }

       // resume does cond broadcast
       ggml_threadpool_resume_locked(threadpool);
    } else {
       ggml_cond_broadcast(&threadpool->cond);
    }

    ggml_mutex_unlock(&threadpool->mutex);
}

#endif // GGML_USE_OPENMP

static struct ggml_threadpool * ggml_threadpool_new_impl(
    struct ggml_threadpool_params * tpp,
               struct ggml_cgraph * cgraph,
                struct ggml_cplan * cplan) {

    struct ggml_threadpool * threadpool =
        ggml_aligned_malloc(sizeof(struct ggml_threadpool));
    {
        threadpool->cgraph           = cgraph;
        threadpool->cplan            = cplan;
        threadpool->n_graph          = 0;
        threadpool->n_barrier        = 0;
        threadpool->n_barrier_passed = 0;
        threadpool->current_chunk    = 0;
        threadpool->stop             = false;
        threadpool->pause            = tpp->paused;
        threadpool->abort            = -1;
        threadpool->workers          = NULL;
        threadpool->n_threads        = tpp->n_threads;
        threadpool->poll             = tpp->poll;
        threadpool->prio             = tpp->prio;
        threadpool->ec               = GGML_STATUS_SUCCESS;
    }

    // Allocate and init workers state
    const size_t workers_size = sizeof(struct ggml_compute_state) * tpp->n_threads;
    struct ggml_compute_state * workers = ggml_aligned_malloc(workers_size);

    memset(workers, 0, workers_size);
    for (int j = 0; j < tpp->n_threads; j++) {
        workers[j].threadpool = threadpool;
        workers[j].ith        = j;
    }

    threadpool->workers = workers;

#ifdef GGML_USE_OPENMP
    int32_t cpumask_iter = 0;

    // Compute CPU masks for each thread
    for (int j = 0; j < tpp->n_threads; j++) {
        ggml_thread_cpumask_next(tpp->cpumask, workers[j].cpumask, tpp->strict_cpu, &cpumask_iter);
    }
#else // GGML_USE_OPENMP
    ggml_mutex_init(&threadpool->mutex);
    ggml_cond_init(&threadpool->cond);

    // Spin the threads for all workers, and update CPU placements.
    // Place the main thread last (towards the higher numbered CPU cores).

    int32_t cpumask_iter = 0;

    for (int j = 1; j < tpp->n_threads; j++) {
        ggml_thread_cpumask_next(tpp->cpumask, workers[j].cpumask, tpp->strict_cpu, &cpumask_iter);

        int32_t rc = ggml_thread_create(&workers[j].thrd, NULL, ggml_graph_compute_secondary_thread, &workers[j]);
        GGML_ASSERT(rc == 0);
    }

    ggml_thread_cpumask_next(tpp->cpumask, workers[0].cpumask, tpp->strict_cpu, &cpumask_iter);

    if (!threadpool->pause) {
        // Update main thread prio and affinity at the start, otherwise we'll do it in resume
        ggml_thread_apply_priority(threadpool->prio);
        if (ggml_thread_cpumask_is_valid(threadpool->workers[0].cpumask)) {
            ggml_thread_apply_affinity(threadpool->workers[0].cpumask);
        }
    }
#endif // GGML_USE_OPENMP

    return threadpool;
}

struct ggml_threadpool * ggml_threadpool_new(struct ggml_threadpool_params * tpp) {
    return ggml_threadpool_new_impl(tpp, NULL, NULL);
}

enum ggml_status ggml_graph_compute(struct ggml_cgraph * cgraph, struct ggml_cplan * cplan) {
    ggml_cpu_init();

    GGML_ASSERT(cplan);
    GGML_ASSERT(cplan->n_threads > 0);
    GGML_ASSERT(cplan->work_size == 0 || cplan->work_data != NULL);

    int n_threads                               = cplan->n_threads;
    struct ggml_threadpool * threadpool = cplan->threadpool;

    bool disposable_threadpool = false;

    if (threadpool == NULL) {
        //GGML_PRINT_DEBUG("Threadpool is not specified. Will create a disposable threadpool : n_threads %d\n", n_threads);
        disposable_threadpool = true;

        struct ggml_threadpool_params ttp = ggml_threadpool_params_default(n_threads);
        threadpool = ggml_threadpool_new_impl(&ttp, cgraph, cplan);
    } else {
        // Reset some of the parameters that need resetting
        // No worker threads should be accessing the parameters below at this stage
        threadpool->cgraph           = cgraph;
        threadpool->cplan            = cplan;
        threadpool->current_chunk    = 0;
        threadpool->abort            = -1;
        threadpool->ec               = GGML_STATUS_SUCCESS;
    }

#ifdef GGML_USE_OPENMP
    if (n_threads > 1) {
        #pragma omp parallel num_threads(n_threads)
        {
            #pragma omp single
            {
                // update the number of threads from the actual number of threads that we got from OpenMP
                n_threads = omp_get_num_threads();
                atomic_store_explicit(&threadpool->n_graph, n_threads, memory_order_relaxed);
            }

            // Apply thread CPU mask and priority
            int ith = omp_get_thread_num();

            ggml_thread_apply_priority(threadpool->prio);
            if (ggml_thread_cpumask_is_valid(threadpool->workers[ith].cpumask)) {
                ggml_thread_apply_affinity(threadpool->workers[ith].cpumask);
            }
            ggml_graph_compute_thread(&threadpool->workers[ith]);
        }
    } else {
        atomic_store_explicit(&threadpool->n_graph, 1, memory_order_relaxed);
        ggml_graph_compute_thread(&threadpool->workers[0]);
    }
#else
    if (n_threads > threadpool->n_threads) {
        GGML_LOG_WARN("cplan requested more threads (%d) than available (%d)\n", n_threads, threadpool->n_threads);
        n_threads = threadpool->n_threads;
    }

    // Kick all threads to start the new graph
    ggml_graph_compute_kickoff(threadpool, n_threads);

    // This is a work thread too
    ggml_graph_compute_thread(&threadpool->workers[0]);
#endif

    // don't leave affinity set on the main thread
    clear_numa_thread_affinity();

    enum ggml_status ret = threadpool->ec;

    if (disposable_threadpool) {
        ggml_threadpool_free(threadpool);
    }

    return ret;
}

enum ggml_status ggml_graph_compute_with_ctx(struct ggml_context * ctx, struct ggml_cgraph * cgraph, int n_threads) {
    struct ggml_cplan cplan = ggml_graph_plan(cgraph, n_threads, NULL);

    cplan.work_data = (uint8_t *)ggml_new_buffer(ctx, cplan.work_size);

    return ggml_graph_compute(cgraph, &cplan);
}

void ggml_cpu_fp32_to_fp32(const float * x, float * y, int64_t n) {
    memcpy(y, x, n * sizeof(float));
}

void ggml_cpu_fp32_to_fp16(const float * x, ggml_fp16_t * y, int64_t n) {
    int64_t i = 0;
#if defined(__F16C__)
#if defined(__AVX512F__)
    for (; i + 15 < n; i += 16) {
        __m512 x_vec = _mm512_loadu_ps(x + i);
        __m256i y_vec = _mm512_cvtps_ph(x_vec, _MM_FROUND_TO_NEAREST_INT);
        _mm256_storeu_si256((__m256i *)(y + i), y_vec);
    }
#endif
    for (; i + 7 < n; i += 8) {
        __m256 x_vec = _mm256_loadu_ps(x + i);
        __m128i y_vec = _mm256_cvtps_ph(x_vec, _MM_FROUND_TO_NEAREST_INT);
        _mm_storeu_si128((__m128i *)(y + i), y_vec);
    }
    for (; i + 3 < n; i += 4) {
        __m128 x_vec = _mm_loadu_ps(x + i);
        __m128i y_vec = _mm_cvtps_ph(x_vec, _MM_FROUND_TO_NEAREST_INT);
        _mm_storel_epi64((__m128i *)(y + i), y_vec);
    }
#elif defined(__riscv_zvfh)
    for (int vl; i < n; i += vl) {
        vl = __riscv_vsetvl_e32m2(n - i);
        vfloat32m2_t vx = __riscv_vle32_v_f32m2(&x[i], vl);
        vfloat16m1_t vy = __riscv_vfncvt_f_f_w_f16m1(vx, vl);
        __riscv_vse16_v_f16m1((_Float16 *)&y[i], vy, vl);
    }
#endif
    for (; i < n; ++i) {
        y[i] = GGML_CPU_FP32_TO_FP16(x[i]);
    }
}

void ggml_cpu_fp16_to_fp32(const ggml_fp16_t * x, float * y, int64_t n) {
    int64_t i = 0;
#if defined(__F16C__)
#if defined(__AVX512F__)
    for (; i + 15 < n; i += 16) {
        __m256i x_vec = _mm256_loadu_si256((const __m256i *)(x + i));
        __m512 y_vec = _mm512_cvtph_ps(x_vec);
        _mm512_storeu_ps(y + i, y_vec);
    }
#endif
    for (; i + 7 < n; i += 8) {
        __m128i x_vec = _mm_loadu_si128((const __m128i *)(x + i));
        __m256 y_vec = _mm256_cvtph_ps(x_vec);
        _mm256_storeu_ps(y + i, y_vec);
    }
    for (; i + 3 < n; i += 4) {
        __m128i x_vec = _mm_loadl_epi64((const __m128i *)(x + i));
        __m128 y_vec = _mm_cvtph_ps(x_vec);
        _mm_storeu_ps(y + i, y_vec);
    }

#elif defined(__riscv_v_intrinsic) && defined(__riscv_zvfhmin)
    // calculate step size
    const int epr = __riscv_vsetvlmax_e16m2();
    const int step = epr * 2;
    const int np = (n & ~(step - 1));

    // unroll by 2
    for (; i < np; i += step) {
        vfloat16m2_t ax0 = __riscv_vle16_v_f16m2((const _Float16*)x + i, epr);
        vfloat32m4_t ay0 = __riscv_vfwcvt_f_f_v_f32m4(ax0, epr);
        __riscv_vse32_v_f32m4(y + i, ay0, epr);

        vfloat16m2_t ax1 = __riscv_vle16_v_f16m2((const _Float16*)x + i + epr, epr);
        vfloat32m4_t ay1 = __riscv_vfwcvt_f_f_v_f32m4(ax1, epr);
        __riscv_vse32_v_f32m4(y + i + epr, ay1, epr);
    }

    // leftovers
    int vl;
    for (i = np; i < n; i += vl) {
        vl = __riscv_vsetvl_e16m2(n - i);
        vfloat16m2_t ax0 = __riscv_vle16_v_f16m2((const _Float16*)x + i, vl);
        vfloat32m4_t ay0 = __riscv_vfwcvt_f_f_v_f32m4(ax0, vl);
        __riscv_vse32_v_f32m4(y + i, ay0, vl);
    }

#endif

    for (; i < n; ++i) {
        y[i] = GGML_CPU_FP16_TO_FP32(x[i]);
    }
}

void ggml_cpu_fp32_to_bf16(const float * x, ggml_bf16_t * y, int64_t n) {
    int64_t i = 0;
    for (; i < n; ++i) {
        y[i] = GGML_FP32_TO_BF16(x[i]);
    }
}

void ggml_cpu_fp32_to_i32(const float * x, int32_t * y, int64_t n) {
    int64_t i = 0;
    for (; i < n; ++i) {
        y[i] = x[i];
    }
}

void ggml_cpu_bf16_to_fp32(const ggml_bf16_t * x, float * y, int64_t n) {
    int64_t i = 0;
#if defined(__AVX2__)
#if defined(__AVX512F__)
    for (; i + 15 < n; i += 16) {
        _mm512_storeu_ps(y + i,
                        _mm512_castsi512_ps(
                            _mm512_slli_epi32(
                                _mm512_cvtepu16_epi32(
                                    _mm256_loadu_si256(
                                        (const __m256i *)(x + i))),
                                16)));
    }
#endif
    for (; i + 7 < n; i += 8) {
        _mm256_storeu_ps(y + i,
                        _mm256_castsi256_ps(
                            _mm256_slli_epi32(
                                _mm256_cvtepu16_epi32(
                                    _mm_loadu_si128(
                                        (const __m128i *)(x + i))),
                                16)));
    }
#elif defined(__riscv_v_intrinsic) && defined(__riscv_zvfbfmin)
    // calculate step size
    const int epr = __riscv_vsetvlmax_e16m2();
    const int step = epr * 2;
    const int np = (n & ~(step - 1));

    // unroll by 2
    for (; i < np; i += step) {
        vbfloat16m2_t ax0 = __riscv_vle16_v_bf16m2((const __bf16*)x + i, epr);
        vfloat32m4_t ay0 = __riscv_vfwcvtbf16_f_f_v_f32m4(ax0, epr);
        __riscv_vse32_v_f32m4(y + i, ay0, epr);

        vbfloat16m2_t ax1 = __riscv_vle16_v_bf16m2((const __bf16*)x + i + epr, epr);
        vfloat32m4_t ay1 = __riscv_vfwcvtbf16_f_f_v_f32m4(ax1, epr);
        __riscv_vse32_v_f32m4(y + i + epr, ay1, epr);
    }

    // leftovers
    int vl;
    for (i = np; i < n; i += vl) {
        vl = __riscv_vsetvl_e16m2(n - i);
        vbfloat16m2_t ax0 = __riscv_vle16_v_bf16m2((const __bf16*)x + i, vl);
        vfloat32m4_t ay0 = __riscv_vfwcvtbf16_f_f_v_f32m4(ax0, vl);
        __riscv_vse32_v_f32m4(y + i, ay0, vl);
    }
#endif
    for (; i < n; i++) {
        y[i] = GGML_BF16_TO_FP32(x[i]);
    }
}

int ggml_cpu_has_avx(void) {
#if defined(__AVX__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_avx_vnni(void) {
#if defined(__AVXVNNI__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_avx2(void) {
#if defined(__AVX2__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_avx512(void) {
#if defined(__AVX512F__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_avx512_vbmi(void) {
#if defined(__AVX512VBMI__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_avx512_vnni(void) {
#if defined(__AVX512VNNI__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_avx512_bf16(void) {
#if defined(__AVX512BF16__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_amx_int8(void) {
#if defined(__AMX_INT8__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_bmi2(void) {
#if defined(__BMI2__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_fma(void) {
#if defined(__FMA__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_arm_fma(void) {
#if defined(__ARM_FEATURE_FMA)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_riscv_v(void) {
#if defined(__riscv_v_intrinsic)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_get_rvv_vlen(void) {
#if defined(__riscv) && defined(__riscv_v_intrinsic)
    return ggml_riscv_arch_features.rvv_vlen;
#else
    return 0;
#endif
}

int ggml_cpu_has_f16c(void) {
#if defined(__F16C__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_fp16_va(void) {
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_wasm_simd(void) {
#if defined(__wasm_simd128__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_llamafile(void) {
#if defined(GGML_USE_LLAMAFILE)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_sse3(void) {
#if defined(__SSE3__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_ssse3(void) {
#if defined(__SSSE3__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_vsx(void) {
#if defined(__POWER9_VECTOR__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_vxe(void) {
#if defined(__VXE__) || defined(__VXE2__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_neon(void) {
#if defined(__ARM_ARCH) && defined(__ARM_NEON)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_dotprod(void) {
#if defined(__ARM_ARCH) && defined(__ARM_FEATURE_DOTPROD)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_sve(void) {
#if defined(__ARM_ARCH) && defined(__ARM_FEATURE_SVE)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_matmul_int8(void) {
#if defined(__ARM_ARCH) && defined(__ARM_FEATURE_MATMUL_INT8)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_get_sve_cnt(void) {
#if defined(__ARM_ARCH) && defined(__ARM_FEATURE_SVE)
    return ggml_arm_arch_features.sve_cnt;
#else
    return 0;
#endif
}

int ggml_cpu_has_sme(void) {
#if defined(__ARM_ARCH) && defined(__ARM_FEATURE_SME)
    return 1;
#else
    return 0;
#endif
}

void ggml_cpu_init(void) {
    // needed to initialize ggml_time
    {
        struct ggml_init_params params = { 0, NULL, false };
        struct ggml_context * ctx = ggml_init(params);
        ggml_free(ctx);
    }

    ggml_critical_section_start();

    static bool is_first_call = true;

    if (is_first_call) {
        // initialize GELU, Quick GELU, SILU and EXP F32 tables
        {
            const uint64_t t_start = ggml_time_us(); UNUSED(t_start);

            for (int i = 0; i < (1 << 16); ++i) {
                union {
                    uint16_t u16;
                    ggml_fp16_t fp16;
                } u = {i};
                float f = GGML_COMPUTE_FP16_TO_FP32(u.fp16);
                ggml_table_f32_f16[i] = f;
                ggml_table_gelu_f16[i] = GGML_CPU_FP32_TO_FP16(ggml_gelu_f32(f));
                ggml_table_gelu_quick_f16[i] = GGML_CPU_FP32_TO_FP16(ggml_gelu_quick_f32(f));
            }

            // initialize E8M0 half table (256 entries)
            for (int i = 0; i < (1 << 8); ++i) {
                ggml_table_f32_e8m0_half[i] = GGML_E8M0_TO_FP32_HALF(i);
            }

            // initialize UE4M3 table (256 entries)
            for (int i = 0; i < (1 << 8); ++i) {
                ggml_table_f32_ue4m3[i] = ggml_ue4m3_to_fp32(i);
            }

            const uint64_t t_end = ggml_time_us(); UNUSED(t_end);

            GGML_PRINT_DEBUG("%s: GELU, Quick GELU, SILU and EXP tables initialized in %f ms\n", __func__, (t_end - t_start)/1000.0);

#ifdef GGML_USE_OPENMP
            //if (!getenv("OMP_WAIT_POLICY")) {
            //    // set the wait policy to active, so that OpenMP threads don't sleep
            //    setenv("OMP_WAIT_POLICY", "active", 0)
            //}

            if (!getenv("KMP_BLOCKTIME")) {
                // set the time to wait before sleeping a thread
                // this is less aggressive than setting the wait policy to active, but should achieve similar results in most cases
#ifdef _WIN32
                _putenv_s("KMP_BLOCKTIME", "200"); // 200ms
#else
                setenv("KMP_BLOCKTIME", "200", 0); // 200ms
#endif
            }
#endif
        }

#if defined(__ARM_ARCH)
        ggml_init_arm_arch_features();
#endif

#if defined(__riscv)
        ggml_init_riscv_arch_features();
#endif

        {
            const char * env = getenv("GGML_CPU_DISABLE_FUSION");
            ggml_cpu_disable_fusion = (env != NULL && atoi(env) == 1);
        }

        // INF-70 SYNC-12: opt-in ggml-cpu fusion passes, default OFF
        {
            const char * env = getenv("GGML_CPU_FUSE_GDN_STATE");
            ggml_cpu_fuse_gdn_state = (env != NULL && atoi(env) == 1);
        }
        {
            const char * env = getenv("GGML_CPU_FUSE_EMPTY");
            ggml_cpu_fuse_empty = (env != NULL && atoi(env) == 1);
        }
        {
            const char * env = getenv("GGML_CPU_FUSE_DEBUG");
            ggml_cpu_fuse_debug = (env != NULL && atoi(env) == 1);
        }

        is_first_call = false;
    }

    ggml_critical_section_end();
}
