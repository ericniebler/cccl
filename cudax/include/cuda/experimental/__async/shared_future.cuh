//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_ASYNC_SHARED_FUTURE__
#define __CUDAX_ASYNC_SHARED_FUTURE__

#include <cuda/__cccl_config>

#include "cuda/std/__cccl/attributes.h"

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/utility>

#include <cuda/experimental/__async/basic_future.cuh>
#include <cuda/experimental/__async/stream_ref.cuh>
#include <cuda/experimental/__event/event.cuh>

#include <memory> // for ::std::shared_ptr

namespace cuda::experimental
{
///
/// shared_future
///
template <class _Value>
struct shared_future : __basic_future<shared_future<_Value>>
{
  using value_type _CCCL_NODEBUG_ALIAS = _Value;

  shared_future(shared_future&&)            = default;
  shared_future& operator=(shared_future&&) = default;

  shared_future(shared_future const&)            = default;
  shared_future& operator=(shared_future const&) = default;

  // Make a ready shared_future:
  _CUDAX_HOST_API shared_future(_Value&& __val)
      : __basic_future<shared_future>(stream_ref(detail::__invalid_stream))
      , __control_block_(::std::make_shared<__control_block>(this->get_stream(), _CUDA_VSTD::move(__val)))
  {}

  event_ref get_event() const noexcept
  {
    return __control_block_->__event_;
  }

  _CUDAX_HOST_API auto get_unsynchronized() && noexcept -> _Value&&
  {
    return _CUDA_VSTD::forward<_Value>(__control_block_->__value_);
  }

  _CUDAX_HOST_API auto get_unsynchronized() & noexcept -> _Value&
  {
    return __control_block_->__value_;
  }

  _CUDAX_HOST_API auto get_unsynchronized() const& noexcept -> _Value const&
  {
    return __control_block_->__value_;
  }

  _CUDAX_HOST_API auto operator->() const noexcept -> _Value const&
  {
    return __control_block_->__value_;
  }

  //! @brief If `valid() ==  false`, no-op; otherwise, calls
  //! cudaEventSynchronize on the result of `get_event()`.
  //! @post `valid() == false`
  //! @remark This function is called from `cudax::wait` and `cudax::wait_all`.
  _CUDAX_HOST_API void wait() const
  {
    auto __stream = __future_access::__exchange_stream(*this, stream_ref(detail::__invalid_stream));
    __control_block_->__event_.wait();
  }

private:
  friend struct __future_access;
  using __basic_future<shared_future>::wait;

  struct __control_block
  {
    __control_block(stream_ref __stream, _Value&& __value)
        : __event_(__stream)
        , __value_(_CUDA_VSTD::forward<_Value>(__value))
    {}

    event __event_;
    _Value __value_;
  };

  _CCCL_TEMPLATE(class _Task)
  _CCCL_REQUIRES(Task<_Task>)
  _CUDAX_HOST_API explicit shared_future(_Task __task, stream_ref __stream)
      : __basic_future<shared_future>(__stream)
      , __control_block_(
          ::std::make_shared<__control_block>(this->get_stream(), _CUDA_VSTD::move(__task).enqueue(__stream)))
  {}

  _CUDAX_HOST_API explicit shared_future(_Value&& __val, stream_ref __stream)
      : __basic_future<shared_future>(__stream)
      , __control_block_(::std::make_shared<__control_block>(this->get_stream(), _CUDA_VSTD::forward<_Value>(__val)))
  {}

  _CUDAX_HOST_API explicit shared_future(future<_Value>&& __fut)
      : __basic_future<shared_future>(__future_access::__exchange_stream(__fut, stream_ref(detail::__invalid_stream)))
      , __control_block_(
          ::std::make_shared<__control_block>(this->get_stream(), _CUDA_VSTD::move(__fut).get_unsynchronized()))
  {}

  ::std::shared_ptr<__control_block> __control_block_;
};

struct __shared_manipulator
{
  template <class _Value>
  _CCCL_NODISCARD_FRIEND _CUDAX_HOST_API auto operator<<(future<_Value>&& __fut, __shared_manipulator)
  {
    return __future_access::__make_shared_future(_CUDA_VSTD::move(__fut));
  }
};

inline constexpr __shared_manipulator shared{};

} // namespace cuda::experimental

#endif // __CUDAX_ASYNC_SHARED_FUTURE__
