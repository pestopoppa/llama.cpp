// iqk port: ik_llama-specific GGML_TYPE_* enum values (repacked _R*, IQ SOTA,
// trellis, etc.) referenced by the kquants/legacy kernels' dispatch as DEAD
// paths (our dispatch only routes Q4_K/Q5_K/Q6_K/Q8_0 x Q8_2_X4). Provided as
// #defines so the kernels COMPILE without polluting v6's ggml_type enum/type_traits.
// Auto-extracted from ik_llama ggml/include/ggml.h.
#pragma once
#ifndef GGML_TYPE_BF16_R16
#define GGML_TYPE_BF16_R16 ((ggml_type)230)
#endif
#ifndef GGML_TYPE_IQ1_BN
#define GGML_TYPE_IQ1_BN ((ggml_type)134)
#endif
#ifndef GGML_TYPE_IQ1_KT
#define GGML_TYPE_IQ1_KT ((ggml_type)158)
#endif
#ifndef GGML_TYPE_IQ1_M_R4
#define GGML_TYPE_IQ1_M_R4 ((ggml_type)229)
#endif
#ifndef GGML_TYPE_IQ1_S_R4
#define GGML_TYPE_IQ1_S_R4 ((ggml_type)219)
#endif
#ifndef GGML_TYPE_IQ2_BN
#define GGML_TYPE_IQ2_BN ((ggml_type)135)
#endif
#ifndef GGML_TYPE_IQ2_BN_R4
#define GGML_TYPE_IQ2_BN_R4 ((ggml_type)335)
#endif
#ifndef GGML_TYPE_IQ2_K
#define GGML_TYPE_IQ2_K ((ggml_type)137)
#endif
#ifndef GGML_TYPE_IQ2_KL
#define GGML_TYPE_IQ2_KL ((ggml_type)157)
#endif
#ifndef GGML_TYPE_IQ2_KS
#define GGML_TYPE_IQ2_KS ((ggml_type)145)
#endif
#ifndef GGML_TYPE_IQ2_KT
#define GGML_TYPE_IQ2_KT ((ggml_type)153)
#endif
#ifndef GGML_TYPE_IQ2_K_R4
#define GGML_TYPE_IQ2_K_R4 ((ggml_type)337)
#endif
#ifndef GGML_TYPE_IQ2_S_R4
#define GGML_TYPE_IQ2_S_R4 ((ggml_type)222)
#endif
#ifndef GGML_TYPE_IQ2_XS_R4
#define GGML_TYPE_IQ2_XS_R4 ((ggml_type)217)
#endif
#ifndef GGML_TYPE_IQ2_XXS_R4
#define GGML_TYPE_IQ2_XXS_R4 ((ggml_type)216)
#endif
#ifndef GGML_TYPE_IQ3_K
#define GGML_TYPE_IQ3_K ((ggml_type)138)
#endif
#ifndef GGML_TYPE_IQ3_KS
#define GGML_TYPE_IQ3_KS ((ggml_type)156)
#endif
#ifndef GGML_TYPE_IQ3_KT
#define GGML_TYPE_IQ3_KT ((ggml_type)154)
#endif
#ifndef GGML_TYPE_IQ3_K_R4
#define GGML_TYPE_IQ3_K_R4 ((ggml_type)338)
#endif
#ifndef GGML_TYPE_IQ3_S_R4
#define GGML_TYPE_IQ3_S_R4 ((ggml_type)221)
#endif
#ifndef GGML_TYPE_IQ3_XXS_R4
#define GGML_TYPE_IQ3_XXS_R4 ((ggml_type)218)
#endif
#ifndef GGML_TYPE_IQ4_K
#define GGML_TYPE_IQ4_K ((ggml_type)139)
#endif
#ifndef GGML_TYPE_IQ4_KS
#define GGML_TYPE_IQ4_KS ((ggml_type)144)
#endif
#ifndef GGML_TYPE_IQ4_KSS
#define GGML_TYPE_IQ4_KSS ((ggml_type)146)
#endif
#ifndef GGML_TYPE_IQ4_KS_R4
#define GGML_TYPE_IQ4_KS_R4 ((ggml_type)344)
#endif
#ifndef GGML_TYPE_IQ4_KT
#define GGML_TYPE_IQ4_KT ((ggml_type)155)
#endif
#ifndef GGML_TYPE_IQ4_K_R4
#define GGML_TYPE_IQ4_K_R4 ((ggml_type)339)
#endif
#ifndef GGML_TYPE_IQ4_NL_R4
#define GGML_TYPE_IQ4_NL_R4 ((ggml_type)220)
#endif
#ifndef GGML_TYPE_IQ4_XS_R8
#define GGML_TYPE_IQ4_XS_R8 ((ggml_type)223)
#endif
#ifndef GGML_TYPE_IQ5_K
#define GGML_TYPE_IQ5_K ((ggml_type)140)
#endif
#ifndef GGML_TYPE_IQ5_KS
#define GGML_TYPE_IQ5_KS ((ggml_type)152)
#endif
#ifndef GGML_TYPE_IQ5_KS_R4
#define GGML_TYPE_IQ5_KS_R4 ((ggml_type)352)
#endif
#ifndef GGML_TYPE_IQ5_K_R4
#define GGML_TYPE_IQ5_K_R4 ((ggml_type)340)
#endif
#ifndef GGML_TYPE_IQ6_K
#define GGML_TYPE_IQ6_K ((ggml_type)141)
#endif
#ifndef GGML_TYPE_Q1_0_G128
#define GGML_TYPE_Q1_0_G128 ((ggml_type)41)
#endif
#ifndef GGML_TYPE_Q2_K_R4
#define GGML_TYPE_Q2_K_R4 ((ggml_type)210)
#endif
#ifndef GGML_TYPE_Q3_K_R4
#define GGML_TYPE_Q3_K_R4 ((ggml_type)211)
#endif
#ifndef GGML_TYPE_Q4_0_R8
#define GGML_TYPE_Q4_0_R8 ((ggml_type)202)
#endif
#ifndef GGML_TYPE_Q4_K_R4
#define GGML_TYPE_Q4_K_R4 ((ggml_type)212)
#endif
#ifndef GGML_TYPE_Q5_0_R4
#define GGML_TYPE_Q5_0_R4 ((ggml_type)206)
#endif
#ifndef GGML_TYPE_Q5_K_R4
#define GGML_TYPE_Q5_K_R4 ((ggml_type)213)
#endif
#ifndef GGML_TYPE_Q6_0
#define GGML_TYPE_Q6_0 ((ggml_type)133)
#endif
#ifndef GGML_TYPE_Q6_0_R4
#define GGML_TYPE_Q6_0_R4 ((ggml_type)233)
#endif
#ifndef GGML_TYPE_Q6_K_R4
#define GGML_TYPE_Q6_K_R4 ((ggml_type)214)
#endif
#ifndef GGML_TYPE_Q8_0_R8
#define GGML_TYPE_Q8_0_R8 ((ggml_type)208)
#endif
#ifndef GGML_TYPE_Q8_K32
#define GGML_TYPE_Q8_K32 ((ggml_type)148)
#endif
#ifndef GGML_TYPE_Q8_KV
#define GGML_TYPE_Q8_KV ((ggml_type)151)
#endif
#ifndef GGML_TYPE_Q8_KV_R8
#define GGML_TYPE_Q8_KV_R8 ((ggml_type)398)
#endif
#ifndef GGML_TYPE_Q8_K_R16
#define GGML_TYPE_Q8_K_R16 ((ggml_type)397)
#endif
#ifndef GGML_TYPE_Q8_K_R8
#define GGML_TYPE_Q8_K_R8 ((ggml_type)399)
#endif

// iqk port: ik uses GGML_UNARY_OP_SWIGLU_OAI; v6 moved it to the GLU-op enum
// (GGML_GLU_OP_SWIGLU_OAI). The unary-op reference is in iqk's fused-up-gate path,
// dead in Stage 1. Provide a sentinel so it compiles (never matched at runtime).
#ifndef GGML_UNARY_OP_SWIGLU_OAI
#define GGML_UNARY_OP_SWIGLU_OAI 9999
#endif
