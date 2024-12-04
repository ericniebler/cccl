//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_ASYNC_DETAIL_GET_UNSYNCHRONIZED
#define __CUDAX_ASYNC_DETAIL_GET_UNSYNCHRONIZED

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/is_valid_expansion.h>
#include <cuda/std/tuple>
#include <cuda/std/utility>

#include <cuda/experimental/__async/async_fwd.cuh>
#include <cuda/experimental/__async/sender/tuple.cuh>
#include <cuda/experimental/__detail/config.cuh>
#include <cuda/experimental/__detail/utility.cuh>

namespace cuda::experimental
{
namespace __get_unsync_ns
{
_CUDAX_HOST_API void cuda_get_unsynchronized();

// The get_unsynchronized customization point is a way for futures and future-like
// types to permit access to its underlying value without synchronizing
// the stream first.
struct __fn
{
  template <class _Future>
  using __result_t = decltype(cuda_get_unsynchronized(declval<_Future>()));

  template <class _Value>
  _CUDAX_TRIVIAL_HOST_API decltype(auto) operator()(_Value&& __value) const
  {
    if constexpr (_CUDA_VSTD::_IsValidExpansion<__result_t, _Value>::value)
    {
      return cuda_get_unsynchronized(static_cast<_Value&&>(__value));
    }
    else
    {
      return _Value(static_cast<_Value&&>(__value));
    }
  }
};
} // namespace __get_unsync_ns

inline namespace __cpo
{
inline constexpr __get_unsync_ns::__fn get_unsynchronized{};
} // namespace __cpo

namespace __async
{
inline constexpr auto __cudax_get_unsync_repack_fn = //
  [](auto&&... __values) {
    using __tuple_t = __tuple<__wait_result_t<decltype(__values)>...>;
    return __tuple_t(get_unsynchronized(static_cast<decltype(__values)&&>(__values))...);
  };

// Customize get_unsynchronized for __async::__tuple so that it calls get_unsynchronized on
// each element.
template <class _Is, class... _Values>
_CUDAX_HOST_API auto cuda_get_unsynchronized(__tupl<_Is, _Values...>&& __tupl)
{
  return __tupl.__apply(__cudax_get_unsync_repack_fn, _CUDA_VSTD::move(__tupl));
}

template <class _Is, class... _Values>
_CUDAX_HOST_API auto cuda_get_unsynchronized(__tupl<_Is, _Values...>& __tupl)
{
  return __tupl.__apply(__cudax_get_unsync_repack_fn, __tupl);
}

template <class _Is, class... _Values>
_CUDAX_HOST_API auto cuda_get_unsynchronized(const __tupl<_Is, _Values...>& __tupl)
{
  return __tupl.__apply(__cudax_get_unsync_repack_fn, __tupl);
}
} // namespace __async

} // namespace cuda::experimental

_LIBCUDACXX_BEGIN_NAMESPACE_STD

inline constexpr auto __cudax_get_unsync_repack_fn = //
  [](auto&&... __values) {
    using __tuple_t = tuple<experimental::__wait_result_t<decltype(__values)>...>;
    return __tuple_t(get_unsynchronized(static_cast<decltype(__values)&&>(__values))...);
  };

// Customize get_unsynchronized for cuda::std::tuple so that it calls get_unsynchronized on
// each element.
template <class... _Values>
_CUDAX_HOST_API auto cuda_get_unsynchronized(tuple<_Values...>&& __tupl)
{
  return _CUDA_VSTD::apply(__cudax_get_unsync_repack_fn, _CUDA_VSTD::move(__tupl));
}

template <class... _Values>
_CUDAX_HOST_API auto cuda_get_unsynchronized(tuple<_Values...>& __tupl)
{
  return _CUDA_VSTD::apply(__cudax_get_unsync_repack_fn, __tupl);
}

template <class... _Values>
_CUDAX_HOST_API auto cuda_get_unsynchronized(const tuple<_Values...>& __tupl)
{
  return _CUDA_VSTD::apply(__cudax_get_unsync_repack_fn, __tupl);
}

_LIBCUDACXX_END_NAMESPACE_STD

#endif // __CUDAX_ASYNC_DETAIL_GET_UNSYNCHRONIZED
