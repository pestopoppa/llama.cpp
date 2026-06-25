//
// Copyright (C) 2024-2025 Iwan Kawrakow
// MIT license
// SPDX-License-Identifier: MIT
//

#pragma once

// iqk port shim: ik_llama internal activation type ids that are NOT in v6's
// public ggml_type enum (v6 ceiling = 42). Q8_2/Q8_2_X4 are activation-only
// formats produced in the dispatch hook's wdata and consumed by iqk kernels;
// they are never on-disk tensor types, so we keep v6's enum/type_traits intact
// and provide the ids here (matching ik's enum values 98/99).
#ifndef GGML_TYPE_Q8_2
#define GGML_TYPE_Q8_2 98
#endif
#ifndef GGML_TYPE_Q8_2_X4
#define GGML_TYPE_Q8_2_X4 99
#endif

#if defined IQK_IMPLEMENT
#undef IQK_IMPLEMENT
#endif

#if defined __AVX2__ || defined __ARM_FEATURE_DOTPROD
#define IQK_IMPLEMENT
#endif

#ifdef GGML_SHARED
#    if defined(_WIN32) && !defined(__MINGW32__)
#        ifdef GGML_BUILD
#            define IQK_API __declspec(dllexport)
#        else
#            define IQK_API __declspec(dllimport)
#        endif
#    else
#        define IQK_API __attribute__ ((visibility ("default")))
#    endif
#else
#    define IQK_API
#endif

#ifdef _MSC_VER
#define IQK_NOINLINE __declspec(noinline)
#define IQK_ALWAYS_INLINE inline
#if !defined __x86_64__ && defined _M_X64
#define __x86_64__
#endif
#else
#define IQK_NOINLINE __attribute__((__noinline__))
#define IQK_ALWAYS_INLINE __attribute__((__always_inline__))
#endif

#if defined __x86_64__
#if defined HAVE_FANCY_SIMD
    #undef HAVE_FANCY_SIMD
#endif
#if defined(__AVX512F__) && defined(__AVX512VNNI__) && defined(__AVX512VL__) && defined(__AVX512BW__) && defined(__AVX512DQ__)
    #define HAVE_FANCY_SIMD
#endif
#if defined HAVE_VNNI256
    #undef HAVE_VNNI256
#endif
#if defined(__AVXVNNI__) || (defined(__AVX512VNNI__) && defined(__AVX512VL__))
    #define HAVE_VNNI256
#endif
#endif

