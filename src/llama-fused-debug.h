#pragma once

// INF-70 A3 — the fused decoder's correctness-hunt instrumentation.
//
// The fused decode path accumulated ~114 fprintf/fopen and ~70 getenv sites
// during the INF-67 bisection, several of them inside the per-expert row loops
// (~2.5 M getenv calls per token) and a few unconditional. They are diagnostic
// only, and they are on the timed path, so every one of them is now behind this
// compile-time switch, which is OFF by default.
//
//   cmake -DCMAKE_CXX_FLAGS=-DQWEN4EXP_FUSED_DEBUG ...   brings them back
//
// The bodies still compile in both configurations (`if (false) { ... }` is
// parsed and type-checked, then eliminated), so the instrumentation cannot rot.
//
// NOT under this switch, deliberately: the GGML_FUSED_PROF per-token profiler
// (Axis A's gate is measured with it) and the fused hook's own opt-in env read.

#include <cstdlib>

#ifdef QWEN4EXP_FUSED_DEBUG
#  define FUSED_DBG(env) (getenv(env) != NULL)
#  define FUSED_DBG_ON   1
#else
#  define FUSED_DBG(env) false
#  define FUSED_DBG_ON   0
#endif
