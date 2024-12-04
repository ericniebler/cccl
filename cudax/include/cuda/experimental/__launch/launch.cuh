//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX__LAUNCH_LAUNCH
#define _CUDAX__LAUNCH_LAUNCH

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda_runtime.h>

#include <cuda/std/__exception/cuda_error.h>
#include <cuda/std/__type_traits/decay.h>
#include <cuda/std/tuple>
#include <cuda/std/utility>
#include <cuda/stream_ref>

#include <cuda/experimental/__algorithm/common.cuh>
#include <cuda/experimental/__async/async_fwd.cuh>
#include <cuda/experimental/__async/async_transform.cuh>
#include <cuda/experimental/__async/get_unsynchronized.cuh>
#include <cuda/experimental/__async/sender/tuple.cuh>
#include <cuda/experimental/__launch/configuration.cuh>
#include <cuda/experimental/__utility/ensure_current_device.cuh>

#if _CCCL_STD_VER >= 2017
namespace cuda::experimental
{

namespace detail
{
template <typename Config, typename Kernel, typename... Args>
__global__ void kernel_launcher(const Config conf, Kernel kernel_fn, Args... args)
{
  kernel_fn(conf, args...);
}

template <typename Kernel, typename... Args>
__global__ void kernel_launcher_no_config(Kernel kernel_fn, Args... args)
{
  kernel_fn(args...);
}

template <typename Config, typename Kernel, typename... Args>
_CCCL_NODISCARD cudaError_t
launch_impl(::cuda::stream_ref stream, Config conf, const Kernel& kernel_fn, const Args&... args)
{
  static_assert(!::cuda::std::is_same_v<decltype(conf.dims), uninit_t>,
                "Can't launch a configuration without hierarchy dimensions");
  cudaLaunchConfig_t config{};
  cudaError_t status                      = cudaSuccess;
  constexpr bool has_cluster_level        = has_level<cluster_level, decltype(conf.dims)>;
  constexpr unsigned int num_attrs_needed = detail::kernel_config_count_attr_space(conf) + has_cluster_level;
  cudaLaunchAttribute attrs[num_attrs_needed == 0 ? 1 : num_attrs_needed];
  config.attrs    = &attrs[0];
  config.numAttrs = 0;
  config.stream   = stream.get();

  status = detail::apply_kernel_config(conf, config, reinterpret_cast<void*>(kernel_fn));
  if (status != cudaSuccess)
  {
    return status;
  }

  config.blockDim = conf.dims.extents(thread, block);
  config.gridDim  = conf.dims.extents(block, grid);

  if constexpr (has_cluster_level)
  {
    auto cluster_dims                            = conf.dims.extents(block, cluster);
    config.attrs[config.numAttrs].id             = cudaLaunchAttributeClusterDimension;
    config.attrs[config.numAttrs].val.clusterDim = {
      static_cast<unsigned int>(cluster_dims.x),
      static_cast<unsigned int>(cluster_dims.y),
      static_cast<unsigned int>(cluster_dims.z)};
    config.numAttrs++;
  }

  // TODO lower to cudaLaunchKernelExC?
  return cudaLaunchKernelEx(&config, kernel_fn, args...);
}
} // namespace detail

/**
 * @brief Launch a kernel functor with specified configuration and arguments
 *
 * Launches a kernel functor object on the specified stream and with specified configuration.
 * Kernel functor object is a type with __device__ operator().
 * Functor might or might not accept the configuration as its first argument.
 *
 *
 * @par Snippet
 * @code
 * #include <cstdio>
 * #include <cuda/experimental/launch.cuh>
 *
 * struct kernel {
 *     template <typename Configuration>
 *     __device__ void operator()(Configuration conf, unsigned int thread_to_print) {
 *         if (conf.dims.rank(cudax::thread, cudax::grid) == thread_to_print) {
 *             printf("Hello from the GPU\n");
 *         }
 *     }
 * };
 *
 * void launch_kernel(cuda::stream_ref stream) {
 *     auto dims    = cudax::make_hierarchy(cudax::block_dims<128>(), cudax::grid_dims(4));
 *     auto config = cudax::make_config(dims, cudax::launch_cooperative());
 *
 *     cudax::launch(stream, config, kernel(), 42);
 * }
 * @endcode
 *
 * @param stream
 * cuda::stream_ref to launch the kernel into
 *
 * @param conf
 * configuration for this launch
 *
 * @param kernel
 * kernel functor to be launched
 *
 * @param args
 * arguments to be passed into the kernel functor
 */
template <typename... Args, typename... Config, typename Dimensions, typename Kernel>
void launch(
  ::cuda::stream_ref stream, const kernel_config<Dimensions, Config...>& conf, const Kernel& kernel, Args&&... args)
{
  __ensure_current_device __dev_setter(stream);
  cudaError_t status;
  auto combined = conf.combine_with_default(kernel);
  if constexpr (::cuda::std::
                  is_invocable_v<Kernel, kernel_config<Dimensions, Config...>, relocatable_value_of_t<Args>...>)
  {
    auto launcher = detail::kernel_launcher<decltype(combined), Kernel, relocatable_value_of_t<Args>...>;
    status        = detail::launch_impl(
      stream, //
      combined,
      launcher,
      combined,
      kernel,
      relocatable_value(async_transform(stream, std::forward<Args>(args)))...);
  }
  else
  {
    static_assert(::cuda::std::is_invocable_v<Kernel, relocatable_value_of_t<Args>...>);
    auto launcher = detail::kernel_launcher_no_config<Kernel, relocatable_value_of_t<Args>...>;
    status        = detail::launch_impl(
      stream, //
      combined,
      launcher,
      kernel,
      relocatable_value(async_transform(stream, std::forward<Args>(args)))...);
  }
  if (status != cudaSuccess)
  {
    ::cuda::__throw_cuda_error(status, "Failed to launch a kernel");
  }
}

/**
 * @brief Launch a kernel function with specified configuration and arguments
 *
 * Launches a kernel function on the specified stream and with specified configuration.
 * Kernel function is a function with __global__ annotation.
 * Function might or might not accept the configuration as its first argument.
 *
 *
 * @par Snippet
 * @code
 * #include <cstdio>
 * #include <cuda/experimental/launch.cuh>
 *
 * template <typename Configuration>
 * __global__ void kernel(Configuration conf, unsigned int thread_to_print) {
 *     if (conf.dims.rank(cudax::thread, cudax::grid) == thread_to_print) {
 *         printf("Hello from the GPU\n");
 *     }
 * }
 *
 * void launch_kernel(cuda::stream_ref stream) {
 *     auto dims    = cudax::make_hierarchy(cudax::block_dims<128>(), cudax::grid_dims(4));
 *     auto config = cudax::make_config(dims, cudax::launch_cooperative());
 *
 *     cudax::launch(stream, config, kernel<decltype(config)>, 42);
 * }
 * @endcode
 *
 * @param stream
 * cuda::stream_ref to launch the kernel into
 *
 * @param conf
 * configuration for this launch
 *
 * @param kernel
 * kernel function to be launched
 *
 * @param args
 * arguments to be passed into the kernel function
 */
template <typename... ExpArgs, typename... ActArgs, typename... Config, typename Dimensions>
void launch(::cuda::stream_ref stream,
            const kernel_config<Dimensions, Config...>& conf,
            void (*kernel)(kernel_config<Dimensions, Config...>, ExpArgs...),
            ActArgs&&... args)
{
  __ensure_current_device __dev_setter(stream);
  cudaError_t status = detail::launch_impl(
    stream, //
    conf,
    kernel,
    conf,
    relocatable_value(async_transform(stream, std::forward<ActArgs>(args)))...);

  if (status != cudaSuccess)
  {
    ::cuda::__throw_cuda_error(status, "Failed to launch a kernel");
  }
}

/**
 * @brief Launch a kernel function with specified configuration and arguments
 *
 * Launches a kernel function on the specified stream and with specified configuration.
 * Kernel function is a function with __global__ annotation.
 * Function might or might not accept the configuration as its first argument.
 *
 * @par Snippet
 * @code
 * #include <cstdio>
 * #include <cuda/experimental/launch.cuh>
 *
 * template <typename Configuration>
 * __global__ void kernel(Configuration conf, unsigned int thread_to_print) {
 *     if (conf.dims.rank(cudax::thread, cudax::grid) == thread_to_print) {
 *         printf("Hello from the GPU\n");
 *     }
 * }
 *
 * void launch_kernel(cuda::stream_ref stream) {
 *     auto dims    = cudax::make_hierarchy(cudax::block_dims<128>(), cudax::grid_dims(4));
 *     auto config = cudax::make_config(dims, cudax::launch_cooperative());
 *
 *     cudax::launch(stream, config, kernel<decltype(config)>, 42);
 * }
 * @endcode
 *
 * @param stream
 * cuda::stream_ref to launch the kernel into
 *
 * @param conf
 * configuration for this launch
 *
 * @param kernel
 * kernel function to be launched
 *
 * @param args
 * arguments to be passed into the kernel function
 */
template <typename... ExpArgs, typename... ActArgs, typename... Config, typename Dimensions>
void launch(::cuda::stream_ref stream,
            const kernel_config<Dimensions, Config...>& conf,
            void (*kernel)(ExpArgs...),
            ActArgs&&... args)
{
  __ensure_current_device __dev_setter(stream);
  cudaError_t status = detail::launch_impl(
    stream, //
    conf,
    kernel,
    relocatable_value(async_transform(stream, std::forward<ActArgs>(args)))...);

  if (status != cudaSuccess)
  {
    ::cuda::__throw_cuda_error(status, "Failed to launch a kernel");
  }
}

template <typename _Config, typename _Kernel, typename... _Args>
struct __launch_action;

template <typename _Kernel, typename... _Args, typename... Config, typename Dimensions>
auto __launch_async(const kernel_config<Dimensions, Config...>& __dims, _Kernel __kernel_fn, _Args... __args)
{
  using _LaunchAction = __launch_action<kernel_config<Dimensions, Config...>, _Kernel, _Args...>;
  return _LaunchAction{__dims, __kernel_fn, _CUDA_VSTD::move(__args)...};
}

template <typename... _Config, typename _Dimensions, typename _Kernel, typename... _Args>
struct __launch_action<kernel_config<_Dimensions, _Config...>, _Kernel, _Args...>
{
  // By using __async::__tuple as the enqueue return type, we opt-in
  // for stream.insert to return a tuple of futures instead of a future
  // of a tuple.
  using __value_t = __async::__tuple<__wait_result_t<_Args>...>;

  auto enqueue(cuda::stream_ref __stream)
  {
    auto __fn = [__stream, this](_Args&... __args) {
      cuda::experimental::launch(__stream, __config_, __kernel_fn_, __args...);
      return __value_t{{get_unsynchronized(_CUDA_VSTD::move(__args))}...};
    };
    return __args_.__apply(__fn, __args_);
  }

private:
  using __config_t = kernel_config<_Dimensions, _Config...>;

  friend auto __launch_async<>(const __config_t&, _Kernel, _Args...);

  explicit __launch_action(__config_t __config, _Kernel __kernel_fn, _Args... __args)
      : __config_(_CUDA_VSTD::move(__config))
      , __kernel_fn_(_CUDA_VSTD::move(__kernel_fn))
      , __args_{{_CUDA_VSTD::move(__args)}...}
  {}

  __config_t __config_;
  _Kernel __kernel_fn_;
  __async::__tuple<_Args...> __args_;
};

_CCCL_TEMPLATE(typename _Kernel, typename... _Args, typename... Config, typename Dimensions)
_CCCL_REQUIRES((async_param<_Args> && ...))
auto launch(const kernel_config<Dimensions, Config...>& __dims, _Kernel __kernel_fn, _Args&&... __args)
{
  return __launch_async(__dims, __kernel_fn, _CUDA_VSTD::forward<_Args>(__args)...);
}

} // namespace cuda::experimental

#endif // _CCCL_STD_VER >= 2017
#endif // _CUDAX__LAUNCH_LAUNCH
