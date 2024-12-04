/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/**
 * Vector addition: C = A + B.
 *
 * This sample is a very basic sample that implements element by element
 * vector addition. It is the same as the sample illustrating Chapter 2
 * of the programming guide with some additions like error checking.
 */

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>

#include <cuda/std/span>

#include <cuda/experimental/algorithm.cuh>
#include <cuda/experimental/async.cuh>
#include <cuda/experimental/buffer.cuh>
#include <cuda/experimental/launch.cuh>
#include <cuda/experimental/stream.cuh>

#include <cstdio>
#include <vector>

namespace cudax = cuda::experimental;
using cuda::std::span;
using cudax::cref;
using cudax::ref;
using cudax::async::device_buffer;

/**
 * CUDA Kernel Device code
 *
 * Computes the vector addition of A and B into C. The 3 vectors have the same
 * number of elements numElements.
 */
__global__ void vectorAdd(span<const float> A, span<const float> B, span<float> C)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i < A.size())
  {
    C[i] = A[i] + B[i] + 0.0f;
  }
}

// This function accepts as arguments either a device_buffer or a future of a
// device_buffer. It returns a cudax::launch action that, when enqueued on a
// stream, will launch the vectorAdd kernel.
auto addVectors(cudax::cref<device_buffer<float>> in1, //
                cudax::cref<device_buffer<float>> in2, //
                cudax::ref<device_buffer<float>> out)
{
  size_t const numElements = in1.get().size();
  auto config              = cudax::distribute<256>(numElements);
  return cudax::launch(config, vectorAdd, in1, in2, out);
}

/**
 * Host main routine
 */
int main(void)
try
{
  // A CUDA stream on which to execute the vector addition kernel
  cudax::stream stream(cudax::devices[0]);
  cudax::async::device_memory mr;

  // Print the vector length to be used, and compute its size
  int numElements = 50000;
  printf("[Vector addition of %d elements]\n", numElements);

  // Allocate the host vectors
  std::vector<float> A(numElements); // input
  std::vector<float> B(numElements); // input
  std::vector<float> C(numElements); // output

  // Initialize the host input vectors
  auto rnd = [] {
    return rand() / (float) RAND_MAX;
  };
  std::generate_n(A.begin(), A.size(), rnd);
  std::generate_n(B.begin(), B.size(), rnd);

  // Create an action that, when enqueued, asynchronously allocates a device buffer:
  auto new_buffer = mr.new_buffer<float>(numElements);
  // Use that action to create two more actions, both of which will allocate a
  // device buffer and copy the host data to it. When these actions are enqueued
  // onto a stream, they will return a future to the allocated device buffer.
  auto copyA = cudax::copy_bytes(cref(A), new_buffer);
  auto copyB = cudax::copy_bytes(cref(B), new_buffer);

  // Enqueue those two operations and a third that will allocate a buffer for
  // the result. This will return a tuple of device buffer futures.
  auto [DA, DB, DC] = stream << cudax::fuse(std::move(copyA), std::move(copyB), new_buffer);

  // Define the kernel launch parameters
  auto config = cudax::distribute<256>(numElements);

  printf("CUDA kernel launch with %d blocks of %d threads\n",
         config.dims.count(cudax::block, cudax::grid),
         config.dims.count(cudax::thread, cudax::block));

  // Launch the vectorAdd kernel, passing the device buffer futures as
  // arguments. Enqueueing a cudax::launch action returns a tuple of futures to
  // the kernel's arguments, which we use here with cuda::std::tie to move
  // the futures back into their local variables.
  cuda::std::tie(DA, DB, DC) = stream << cudax::launch(config, vectorAdd, std::move(DA), std::move(DB), std::move(DC));

  printf("copying result from device and waiting for it to finish\n");
  // To copy the bytes from the device to the host buffer, we need to transfer ownership
  // of the host buffer to the operation. Enqueueing this copy_bytes action will return
  // a future of the host buffer, which we then wait on before we can access it:
  cudax::future C2 = stream << cudax::copy_bytes(std::move(DC), std::move(C));

  // Wait for the result to be ready. This will block until the kernel has finished
  // and return the host buffer, which is now safe to read.
  C = cudax::wait(std::move(C2));

  printf("verifying the results\n");
  // Verify that the result vector is correct
  for (int i = 0; i < numElements; ++i)
  {
    if (fabs(A[i] + B[i] - C[i]) > 1e-5)
    {
      fprintf(stderr, "Result verification failed at element %d!\n", i);
      exit(EXIT_FAILURE);
    }
  }

  printf("Test PASSED\n");

  printf("Done\n");
  return 0;
}
catch (const std::exception& e)
{
  printf("caught an exception: \"%s\"\n", e.what());
}
catch (...)
{
  printf("caught an unknown exception\n");
}
