//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX___EXECUTION_GRAPH_CONTEXT
#define __CUDAX___EXECUTION_GRAPH_CONTEXT

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/experimental/__execution/graph/context.cuh> // IWYU pragma: export
#include <cuda/experimental/__execution/graph/domain.cuh> // IWYU pragma: export
#include <cuda/experimental/__execution/graph/launch.cuh> // IWYU pragma: export
#include <cuda/experimental/__execution/graph/sync_wait.cuh> // IWYU pragma: export
#include <cuda/experimental/__execution/graph/then.cuh> // IWYU pragma: export

#endif //__CUDAX___EXECUTION_GRAPH_CONTEXT
