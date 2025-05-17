//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX__EXECUTION_STREAM_ADAPTOR
#define __CUDAX__EXECUTION_STREAM_ADAPTOR

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__memory/addressof.h>
#include <cuda/std/__tuple_dir/ignore.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/remove_cvref.h>
#include <cuda/std/__type_traits/remove_reference.h>
#include <cuda/std/__type_traits/type_list.h>
#include <cuda/std/__utility/pod_tuple.h>

#include <cuda/experimental/__detail/utility.cuh>
#include <cuda/experimental/__execution/completion_signatures.cuh>
#include <cuda/experimental/__execution/domain.cuh>
#include <cuda/experimental/__execution/stream/domain.cuh>
#include <cuda/experimental/__execution/variant.cuh>
#include <cuda/experimental/__launch/configuration.cuh>
#include <cuda/experimental/__launch/launch.cuh>
#include <cuda/experimental/__stream/stream_ref.cuh>

#include <nv/target>

#include <cuda_runtime_api.h>

#include <cuda/experimental/__execution/prologue.cuh>

_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_GCC("-Wattributes")

namespace cuda::experimental::execution::__stream
{
template <class _Rcvr>
struct __completion_fn
{
  template <class _Tag, class... _Args>
  _CCCL_DEVICE_API void operator()(_Tag, _Args&&... __args) const noexcept
  {
    _Tag()(static_cast<_Rcvr&&>(__rcvr_), static_cast<_Args&&>(__args)...);
  }

  _Rcvr& __rcvr_;
};

template <class _Rcvr>
struct __results_visitor
{
  template <class _Tuple>
  _CCCL_DEVICE_API void operator()(_Tuple&& __tuple) const noexcept
  {
    _CUDA_VSTD::__apply(__completion_fn<_Rcvr>{__rcvr_}, static_cast<_Tuple&&>(__tuple));
  }

  _Rcvr& __rcvr_;
};

template <class _Rcvr, class _Variant>
struct __state_t
{
  _Rcvr __rcvr_;
  _Variant __results_;
  stream_ref __stream_;
  bool __complete_inline_;
};

template <class _Completions>
_CCCL_API static constexpr auto __with_cuda_error(_Completions __completions) noexcept
{
  return __completions - __eptr_completion() + completion_signatures<set_error_t(cudaError_t)>();
}

template <class _OpState>
__global__ void __host_start_fn(_OpState* __opstate)
{
  __opstate->start();
}

template <class _Rcvr, class _Variant>
__global__ void __host_complete_fn(__state_t<_Rcvr, _Variant>* __state)
{
  _Variant::__visit(__results_visitor<_Rcvr>{__state->__rcvr_}, __state->__results_);
}

template <class _Rcvr, class _Variant>
struct __rcvr_t
{
  template <class _Tag, class... _Args>
  _CCCL_API void __complete(_Tag, _Args&&... __args) noexcept
  {
    if (__state_->__complete_inline_)
    {
      _Tag()(static_cast<_Rcvr&&>(__state_->__rcvr_), static_cast<_Args&&>(__args)...);
    }
    else
    {
      using __tuple_t = _CUDA_VSTD::__decayed_tuple<set_value_t, _Args...>;
      __state_->__results_.template __emplace<__tuple_t>(_Tag(), static_cast<_Args&&>(__args)...);
    }
  }

  template <class... _Args>
  _CCCL_TRIVIAL_API void set_value(_Args&&... __args) noexcept
  {
    __complete(execution::set_value, static_cast<_Args&&>(__args)...);
  }

  template <class _Error>
  _CCCL_TRIVIAL_API void set_error(_Error&& __err) noexcept
  {
    if constexpr (_CUDA_VSTD::_IsSame<_CUDA_VSTD::remove_cvref_t<_Error>, ::std::exception_ptr>::value)
    {
      __complete(execution::set_error, cudaErrorUnknown);
    }
    else
    {
      __complete(execution::set_error, static_cast<_Error&&>(__err));
    }
  }

  _CCCL_TRIVIAL_API void set_stopped() noexcept
  {
    __complete(execution::set_stopped);
  }

  _CCCL_API auto get_env() const noexcept -> env_of_t<_Rcvr>
  {
    return get_env(__state_->__rcvr_);
  }

  __state_t<_Rcvr, _Variant>* __state_;
};

template <class _Tag>
struct __opstate_base_t
{
  _CCCL_API explicit __opstate_base_t(_CUDA_VSTD::__ignore_t) noexcept {}

  _CCCL_API static constexpr auto __get_start_config() noexcept
  {
    return make_config(make_hierarchy(block_dims<1>, grid_dims<1>));
  }
};

// template <>
// struct __opstate_base_t<bulk_t>
// {
//   template <class _CvSndr>
//   _CCCL_API explicit __opstate_base_t(_CvSndr&& __sndr) noexcept
//   {
//     auto&& [__tag, __params, __child] = __sndr;
//     auto&& [__shape, __fn] = __params;
//     __shape_ = __shape;
//   }

//   _CCCL_API static constexpr auto __get_start_config() noexcept
//   {
//     constexpr int __block = 256;
//     const int __grid = (static_cast<int>(__shape_) + __block - 1) / __block;
//     return make_config(make_hierarchy(block_dims<__block>, grid_dims(__grid)));
//   }

//   size_t __shape_{};
// };

template <class _CvSndr, class _Rcvr>
struct __opstate_t : __opstate_base_t<tag_of_t<_CvSndr>>
{
  using operation_state_concept = operation_state_t;

  _CCCL_API explicit __opstate_t(_CvSndr&& __sndr, _Rcvr __rcvr, stream_ref __stream)
      : __opstate_base_t<tag_of_t<_CvSndr>>{__sndr}
      , __state_{static_cast<_Rcvr&&>(__rcvr), {}, __stream, false}
      , __child_opstate_{transform_sender(default_domain(), static_cast<_CvSndr&&>(__sndr), get_env(__state_.__rcvr_))
                           .connect(__rcvr_t{&__state_})}
  {}

  _CCCL_IMMOVABLE_OPSTATE(__opstate_t);

  _CCCL_API void start() noexcept
  {
    NV_IF_TARGET(NV_IS_HOST, (__host_start();), (__device_start();));
  }

  // private:
  using __cv_sndr_t           = __transform_sender_result_t<default_domain, _CvSndr, env_of_t<_Rcvr>>;
  using __sndr_t              = _CUDA_VSTD::remove_reference_t<__cv_sndr_t>;
  using __child_completions_t = decltype(__sndr_t::template get_completion_signatures<_CvSndr, env_of_t<_Rcvr>>());
  using __completions_t       = decltype(__with_cuda_error(__child_completions_t()));
  using __variant_t       = typename __completions_t::template __transform_q<_CUDA_VSTD::__decayed_tuple, __variant>;
  using __rcvr_t          = __stream::__rcvr_t<_Rcvr, __variant_t>;
  using __child_opstate_t = decltype(declval<__cv_sndr_t>().connect(declval<__rcvr_t>()));
  using __start_fn_t      = void(__child_opstate_t*);
  using __complete_fn_t   = void(__state_t<_Rcvr, __variant_t>*);

  _CCCL_HOST_API void __host_start() noexcept
  {
    try
    {
      auto const __stream = __state_.__stream_.get();
      auto const __config = this->__get_start_config();

      cudaPointerAttributes attrs;
      _CCCL_TRY_CUDA_API(cudaPointerGetAttributes, "", &attrs, &__child_opstate_);

      // enqueue a kernel launch to start the child operation state
      experimental::launch(__stream, __config, &__host_start_fn<__child_opstate_t>, &__child_opstate_);
      // BUGBUG the start kernel may finish before the results are ready. Needs synchronization.

      __state_.__stream_.sync(); // BUGBUG

      // enqueue a kernel launch to pass the results to the receiver
      // TODO: this may need a different launch configuration
      experimental::launch(__stream, __config, &__host_complete_fn<_Rcvr, __variant_t>, &__state_);

      __state_.__stream_.sync(); // BUGBUG
    }
    catch (...)
    {
      execution::set_error(static_cast<_Rcvr&&>(__state_.__rcvr_), cudaErrorUnknown);
    }
  }

  _CCCL_DEVICE_API void __device_start() noexcept
  {
    [[maybe_unused]] auto* const __start_fn    = &__host_start_fn<__child_opstate_t>;
    [[maybe_unused]] auto* const __complete_fn = &__host_complete_fn<_Rcvr, __variant_t>;
    __state_.__complete_inline_                = true;
    __child_opstate_.start();
  }

  __state_t<_Rcvr, __variant_t> __state_;
  __child_opstate_t __child_opstate_;
};

template <class _Sndr>
struct __sndr_t
{
  using sender_concept = sender_t;

  template <class _Self, class _Env>
  [[nodiscard]] _CCCL_API static constexpr auto get_completion_signatures() noexcept
  {
    constexpr auto __completions = _Sndr::template get_completion_signatures<__copy_cvref_t<_Self, _Sndr>, _Env>();
    return __with_cuda_error(__completions);
  }

  template <class _Rcvr>
  [[nodiscard]] _CCCL_API auto connect(_Rcvr __rcvr) && -> __opstate_t<_Sndr, _Rcvr>
  {
    return __opstate_t<_Sndr, _Rcvr>(
      static_cast<_Sndr&&>(__state_.__sndr_), static_cast<_Rcvr&&>(__rcvr), __state_.__stream_);
  }

  template <class _Rcvr>
  [[nodiscard]] _CCCL_API auto connect(_Rcvr __rcvr) const& -> __opstate_t<const _Sndr&, _Rcvr>
  {
    return __opstate_t<const _Sndr&, _Rcvr>(__state_.__sndr_, static_cast<_Rcvr&&>(__rcvr), __state_.__stream_);
  }

  [[nodiscard]] _CCCL_API auto get_env() const noexcept -> env_of_t<_Sndr>
  {
    return execution::get_env(__state_.__sndr);
  }

  struct __state_t
  {
    stream_ref __stream_;
    _Sndr __sndr_;
  } __state_;
};

template <class _Sndr, class _Env>
_CCCL_API auto __adapt(_Sndr __sndr, const _Env& __env) -> decltype(auto)
{
  auto __stream = get_stream(env<env_of_t<_Sndr>, const _Env&>{get_env(__sndr), __env});
  return __sndr_t<_Sndr>{{__stream, static_cast<_Sndr&&>(__sndr)}};
}
} // namespace cuda::experimental::execution::__stream

_CCCL_DIAG_POP

#include <cuda/experimental/__execution/epilogue.cuh>

#endif // __CUDAX__EXECUTION_STREAM_LAUNCH
