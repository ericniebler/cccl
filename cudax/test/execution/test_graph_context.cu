//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// Include this first
#include <cuda/experimental/execution.cuh>

// Then include the test helpers
#include <nv/target>

#include "testing.cuh" // IWYU pragma: keep

_CCCL_BEGIN_NV_DIAG_SUPPRESS(177) // function "_is_on_device" was declared but never referenced

namespace task = cuda::experimental::execution;

namespace
{
_CCCL_API bool _is_on_device() noexcept
{
  NV_IF_ELSE_TARGET(NV_IS_HOST, //
                    ({ return false; }),
                    ({ return true; }));
}

void graph_context_test1()
{
  // cudax::stream stream;
  // task::graph_context ctx{stream};
  // auto sched = ctx.get_scheduler();

  // auto sndr = task::schedule(sched) //
  //           | task::then([] __device__() noexcept -> bool {
  //               return _is_on_device();
  //             });

  // auto [on_device] = task::sync_wait(std::move(sndr)).value();
  // CHECK(on_device);
}

C2H_TEST("a simple use of the graph context", "[context][graph]")
{
  // put the test in a separate function to avoid an nvc++ issue with
  // extended lambdas in functions with internal linkage (as is the case
  // with C2H tests).
  graph_context_test1();
}
} // namespace

_CCCL_END_NV_DIAG_SUPPRESS()
