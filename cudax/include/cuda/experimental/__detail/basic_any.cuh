//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_DETAIL_BASIC_ANY_H
#define __CUDAX_DETAIL_BASIC_ANY_H

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__concepts/__concept_macros.h>
#include <cuda/std/__concepts/copyable.h>
#include <cuda/std/__concepts/movable.h>
#include <cuda/std/__concepts/same_as.h>
#include <cuda/std/__functional/invoke.h>
#include <cuda/std/__type_traits/decay.h>
#include <cuda/std/__type_traits/is_base_of.h>
#include <cuda/std/__type_traits/is_valid_expansion.h>
#include <cuda/std/__utility/declval.h>
#include <cuda/std/__utility/exchange.h>

#include <new> // IWYU pragma: keep (needed for placement new)
#include <typeinfo>

namespace cuda::experimental
{
template <class _Interface>
struct __basic_any;

struct _
{};

namespace detail
{
template <class _Ty, class _Uy>
_LIBCUDACXX_CONCEPT __not_derived_from = !_CUDA_VSTD::is_base_of_v<_Uy, _CUDA_VSTD::decay_t<_Ty>>;

template <class _Interface, class _Ty = _Interface>
using __members_of = typename _Interface::template members<_Ty>;

template <class _Ty, class _Interface>
_LIBCUDACXX_CONCEPT __satisfies = _CUDA_VSTD::_IsValidExpansion<__members_of, _Interface, _Ty>::value;

_CCCL_GLOBAL_CONSTANT std::size_t __default_small_buffer_size = 3 * sizeof(void*);

constexpr std::size_t __buffer_size(std::size_t __size) noexcept
{
  return 0 == __size ? __default_small_buffer_size : (__size < sizeof(void*) ? sizeof(void*) : __size);
}

template <std::size_t _Size>
union __storage
{
  alignas(std::max_align_t) unsigned char __buffer_[_Size];
  void* __ptr_;
};

template <class _Ret, class... _As>
using __fn_t = _Ret(_As...);

template <class, auto _Mbr, bool, class = decltype(_Mbr)>
extern int __vfn;

template <class _Ty, auto _Mbr, bool _Tiny, class _Ret, class _Cy, class... _As>
_Ret __vfn_impl(void* __pv, _As... __as)
{
  // __pv is a pointer to a __storage object.
  _Cy* __obj = _Tiny ? static_cast<_Ty*>(__pv) : *static_cast<_Ty* const*>(__pv);
  return _CUDA_VSTD::invoke(_Mbr, *__obj, static_cast<_As&&>(__as)...);
}

template <class _Ty, auto _Mbr, bool _Tiny, class _Ret, class _Cy, class... _As>
_CCCL_GLOBAL_CONSTANT __fn_t<_Ret, void*, _As...>* __vfn<_Ty, _Mbr, _Tiny, _Ret (_Cy::*)(_As...)> =
  &__vfn_impl<_Ty, _Mbr, _Tiny, _Ret, _Cy, _As...>;

template <class _Ty, auto _Mbr, bool _Tiny, class _Ret, class _Cy, class... _As>
_CCCL_GLOBAL_CONSTANT __fn_t<_Ret, void*, _As...>* __vfn<_Ty, _Mbr, _Tiny, _Ret (_Cy::*)(_As...) const> =
  &__vfn_impl<_Ty, _Mbr, _Tiny, _Ret, _Cy const, _As...>;

template <class _Ty, auto _Fn, bool _Tiny, class _Ret, class _Cy, class... _As>
_CCCL_GLOBAL_CONSTANT __fn_t<_Ret, void*, _As...>* __vfn<_Ty, _Fn, _Tiny, _Ret (*)(_Cy&, _As...)> =
  &__vfn_impl<_Ty, _Fn, _Tiny, _Ret, _Cy, _As...>;

template <class _Mbrs>
using __vtable_of = typename _Mbrs::__vtable;

template <class... _VTbls>
struct __vtable : _VTbls...
{};

template <auto _Mbr>
struct __member
{
  using __fn_t = decltype(__vfn<void, _Mbr, true>);
  __fn_t __fn_;
};

template <class _Ty, auto... _Mbrs>
struct __members
{
  using __vtable = detail::__vtable<__member<_Mbrs>...>;

  template <class _ISelf, bool _Tiny>
  static constexpr auto __mk_vtable() noexcept
  {
    return __vtable_of<__members_of<_ISelf>>{{__vfn<_Ty, _Mbrs, _Tiny>}...};
  }
};

template <class _Interface, class _Any>
using __rebind = typename _Interface::template __rebind<_Any>;

template <class _Any, size_t Size = 0, class... _IBases>
struct __interface;

template <template <class...> class _Interface, class _Any, class... _Ts, size_t Size, class... _IBases>
struct __interface<_Interface<_Any, _Ts...>, Size, _IBases...> : __rebind<_IBases, _Any>...
{
  static constexpr size_t __size = __buffer_size(Size);

  using __any_t   = _Any;
  using __iface_t = _Interface<_Any, _Ts...>;

  template <class _NewAny>
  using __rebind = _Interface<_NewAny, _Ts...>;

  template <class _Ty, bool _Tiny>
  static constexpr auto __mk_vtable() noexcept
  {
    return __vtable{detail::__rebind<_IBases, _Any>::template __mk_vtable<_Ty, _Tiny>()...,
                    __members_of<__iface_t, _Ty>::template __mk_vtable<__iface_t, _Tiny>()};
  }
};

template <class _Interface>
using __base_vtable_of = decltype(_Interface::template __mk_vtable<_Interface, true>());

template <class _Interface>
struct __poly_box;

template <class _Interface>
struct __vtable_for : __base_vtable_of<_Interface>
{
  ::std::type_info const* __type_;
  bool __tiny_;
  void* (*__getobj_)(__poly_box<_Interface> const*) noexcept;
  void (*__destroy_)(void*) noexcept;
  void (*__move_)(void*, void*) noexcept;
  void (*__copy_)(void*, void const*);
};

template <class _Ty, bool _Tiny>
void __destroy_(void* __pv) noexcept
{
  if constexpr (_Tiny)
  {
    static_cast<_Ty*>(__pv)->~_Ty();
  }
  else
  {
    delete *static_cast<_Ty* const*>(__pv);
  }
}

template <class _Ty, bool _Tiny>
void __move_(void* __pv, void* __other) noexcept
{
  if constexpr (_CUDA_VSTD::move_constructible<_Ty>)
  {
    if constexpr (_Tiny)
    {
      ::new (__pv) _Ty(static_cast<_Ty&&>(*static_cast<_Ty*>(__other)));
      static_cast<_Ty*>(__other)->~_Ty();
    }
    else
    {
      *static_cast<_Ty**>(__pv) = _CUDA_VSTD::exchange(*static_cast<_Ty**>(__other), nullptr);
    }
  }
}

template <class _Ty, bool _Tiny>
void __copy_(void* __pv, void const* __other)
{
  if constexpr (_CUDA_VSTD::copy_constructible<_Ty>)
  {
    if constexpr (_Tiny)
    {
      ::new (__pv) _Ty(*static_cast<_Ty const*>(__other));
    }
    else
    {
      *static_cast<_Ty**>(__pv) = new _Ty(**static_cast<_Ty const* const*>(__other));
    }
  }
}

template <class _Interface>
struct __poly_box
{
  _LIBCUDACXX_TEMPLATE(class _Obj, class _Ty = _CUDA_VSTD::decay_t<_Obj>)
  _LIBCUDACXX_REQUIRES(__not_derived_from<_Obj, __poly_box> _LIBCUDACXX_AND __satisfies<_Ty, _Interface>)
  __poly_box(_Obj&& __obj) //
    noexcept(noexcept(_Ty(_CUDA_VSTD::declval<_Obj>())) && (sizeof(_Ty) <= _Interface::__size))
      : __vptr_{__vptr_for<_Ty>()}
  {
    if constexpr (sizeof(_Ty) <= _Interface::__size)
    {
      ::new (static_cast<void*>(__obj_.__buffer_)) _Ty(static_cast<_Obj&&>(__obj));
    }
    else
    {
      __obj_.__ptr_ = new _Ty(static_cast<_Obj&&>(__obj));
    }
  }

  _LIBCUDACXX_TEMPLATE(class _Cy = _Interface)
  _LIBCUDACXX_REQUIRES(_CUDA_VSTD::move_constructible<_Cy>)
  __poly_box(__poly_box&& __other) noexcept
  {
    if ((__vptr_ = _CUDA_VSTD::exchange(__other.__vptr_, nullptr)))
    {
      __vptr_->__move_(&__obj_, &__other.__obj_);
    }
  }

  _LIBCUDACXX_TEMPLATE(class _Cy = _Interface)
  _LIBCUDACXX_REQUIRES(_CUDA_VSTD::copy_constructible<_Cy>)
  __poly_box(__poly_box const& __other)
  {
    if ((__vptr_ = __other.__vptr_))
    {
      __vptr_->__copy_(&__obj_, &__other.__obj_);
    }
  }

  ~__poly_box()
  {
    if (__vptr_)
    {
      __vptr_->__destroy_(&__obj_);
    }
  }

  _LIBCUDACXX_TEMPLATE(class _Cy = _Interface)
  _LIBCUDACXX_REQUIRES(_CUDA_VSTD::movable<_Cy>)
  [[maybe_unused]]
  __poly_box& operator=(__poly_box&& __other) noexcept
  {
    if (this != &__other)
    {
      if (__vptr_)
      {
        __vptr_->__destroy_(&__obj_);
        __vptr_ = nullptr;
      }

      if (__other.__vptr_)
      {
        __vptr_ = _CUDA_VSTD::exchange(__other.__vptr_, nullptr);
        __vptr_->__move_(&__obj_, &__other.__obj_);
      }
    }

    return *this;
  }

  _LIBCUDACXX_TEMPLATE(class _Cy = _Interface)
  _LIBCUDACXX_REQUIRES(_CUDA_VSTD::copyable<_Cy>)
  [[maybe_unused]]
  __poly_box& operator=(__poly_box const& __other)
  {
    return this == &__other ? *this : operator=(__poly_box(__other));
  }

  static void* __getobj_(__poly_box const* __base) noexcept
  {
    return const_cast<void*>(static_cast<void const*>(&static_cast<__poly_box<_Interface> const*>(__base)->__obj_));
  }

  template <class _Ty>
  static __vtable_for<_Interface> const* __vptr_for() noexcept
  {
    constexpr bool __tiny = (sizeof(_Ty) <= _Interface::__size);

    static constexpr __vtable_for<_Interface> __s_vtable{
      _Interface::template __mk_vtable<_Ty, __tiny>(),
      &typeid(_Ty),
      __tiny,
      &__getobj_,
      &__destroy_<_Ty, __tiny>,
      &__move_<_Ty, __tiny>,
      &__copy_<_Ty, __tiny>};

    return &__s_vtable;
  }

  template <class _Ty>
  _Ty& __cast() noexcept
  {
    static_assert(_CUDA_VSTD::same_as<_Ty, _CUDA_VSTD::decay_t<_Ty>>);
    constexpr bool __tiny = (sizeof(_Ty) <= _Interface::__size);
    _LIBCUDACXX_ASSERT(typeid(_Ty) == *__vptr_->__type_, "");
    return __tiny ? *static_cast<_Ty*>((void*) &__obj_) : **static_cast<_Ty**>((void*) &__obj_);
  }

  template <class _Ty>
  _Ty const& __cast() const noexcept
  {
    static_assert(_CUDA_VSTD::same_as<_Ty, _CUDA_VSTD::decay_t<_Ty>>);
    constexpr bool __tiny = (sizeof(_Ty) <= _Interface::__size);
    _LIBCUDACXX_ASSERT(typeid(_Ty) == *__vptr_->__type_, "");
    return __tiny ? *static_cast<_Ty const*>((void const*) &__obj_) : **static_cast<_Ty const* const*>((void*) &__obj_);
  }

  __vtable_for<_Interface> const* __vptr_{};
  __storage<_Interface::__size> __obj_{};
};

template <class _Self>
auto __get_poly_box(_Self& __self) noexcept -> decltype(auto)
{
  using __any_t = typename _Self::__any_t;
  using __box_t = typename __any_t::__box_t;
  // C-style cast to skip accessibility check:
  return (__box_t&) static_cast<__any_t&>(__self);
}

template <class _Self>
auto __get_poly_box(_Self const& __self) noexcept -> decltype(auto)
{
  using __any_t = typename _Self::__any_t;
  using __box_t = typename __any_t::__box_t;
  // C-style cast to skip accessibility check:
  return (__box_t const&) static_cast<__any_t const&>(__self);
}

template <auto _Mbr, class _Ret, class _Cy, class... _As>
_Ret __vcall_impl(_Cy& __self, _As... __as)
{
  auto& __box = detail::__get_poly_box(__self);
  auto __fn   = __box.__vptr_->__member<_Mbr>::__fn_;
  return __fn((void*) &__box.__obj_, static_cast<_As&&>(__as)...);
}

template <auto _Mbr, class = decltype(_Mbr)>
extern int __vcall;

// Non-const member functions
template <auto _Mbr, class _Ret, class _Cy, class... _As>
_CCCL_GLOBAL_CONSTANT auto __vcall<_Mbr, _Ret (_Cy::*)(_As...)> = &__vcall_impl<_Mbr, _Ret, _Cy, _As...>;

// Const member functions
template <auto _Mbr, class _Ret, class _Cy, class... _As>
_CCCL_GLOBAL_CONSTANT auto __vcall<_Mbr, _Ret (_Cy::*)(_As...) const> = &__vcall_impl<_Mbr, _Ret, _Cy const, _As...>;

// Non-const free functions
template <auto _Mbr, class _Ret, class _Cy, class... _As>
_CCCL_GLOBAL_CONSTANT auto __vcall<_Mbr, _Ret (*)(_Cy&, _As...)> = &__vcall_impl<_Mbr, _Ret, _Cy, _As...>;

// Const free functions
template <auto _Mbr, class _Ret, class _Cy, class... _As>
_CCCL_GLOBAL_CONSTANT auto __vcall<_Mbr, _Ret (*)(_Cy const&, _As...)> = &__vcall_impl<_Mbr, _Ret, _Cy const, _As...>;

template <class _Self = _>
struct __any_immovable : __interface<__any_immovable<_Self>>
{
  __any_immovable()                                  = default;
  __any_immovable(__any_immovable&&)                 = delete;
  __any_immovable(__any_immovable const&)            = delete;
  __any_immovable& operator=(__any_immovable&&)      = delete;
  __any_immovable& operator=(__any_immovable const&) = delete;

  template <class _Ty>
  using members = __members<_Ty>;
};

template <class _Self = _>
struct __any_moveonly : __interface<__any_moveonly<_Self>>
{
  __any_moveonly()                                 = default;
  __any_moveonly(__any_moveonly&&)                 = default;
  __any_moveonly(__any_moveonly const&)            = delete;
  __any_moveonly& operator=(__any_moveonly&&)      = default;
  __any_moveonly& operator=(__any_moveonly const&) = delete;

  template <class _Ty>
  using members = __members<_Ty>;
};

template <class _Self = _>
struct __any_equality_comparable : __interface<__any_equality_comparable<_Self>>
{
  template <class _Ty>
  static constexpr bool __not_self = !_CUDA_VSTD::same_as<_Ty, __any_equality_comparable>;

  template <class _Ty>
  using __equal_fn_t = __fn_t<bool, _Ty const&, __any_equality_comparable const&>;

  template <class _Ty, _CUDA_VSTD::enable_if_t<__not_self<_Ty>, int> _Enable = 0>
  static auto __equal(_Ty const& __lhs, __any_equality_comparable const& __rhs) -> decltype(__lhs == __lhs)
  {
    auto& __rhs_base = detail::__get_poly_box(__rhs);
    return __lhs == __rhs_base.template __cast<_Ty>();
  }

  static auto __equal(__any_equality_comparable const& __lhs, __any_equality_comparable const& __rhs) -> bool
  {
    return true;
  }

  friend bool operator==(const _Self& __lhs, const _Self& __rhs)
  {
    __any_equality_comparable const& __lhs2 = __lhs;
    __any_equality_comparable const& __rhs2 = __rhs;

    auto& __lhs_box = detail::__get_poly_box(__lhs2);
    auto& __rhs_box = detail::__get_poly_box(__rhs2);

    _LIBCUDACXX_ASSERT(__lhs_box.__vptr_ && __rhs_box.__vptr_, "comparison to moved-from object");

    // Check that the two objects contain the same type:
    if (*__lhs_box.__vptr_->__type_ != *__rhs_box.__vptr_->__type_)
    {
      return false;
    }

    constexpr __equal_fn_t<__any_equality_comparable>* __eq = &__equal;
    return __vcall<__eq>(__lhs2, __rhs2);
  }

  friend bool operator!=(const _Self& __lhs, const _Self& __rhs)
  {
    return !(__lhs == __rhs);
  }

  template <class _Ty>
  using members = __members<_Ty, static_cast<__equal_fn_t<_Ty>*>(&__equal)>;
};

} // namespace detail

template <class _Interface, auto... _Mbrs>
using __members = detail::__members<_Interface, _Mbrs...>;

template <class _ISelf, size_t Size = 0, class... _IBases>
using __interface = detail::__interface<_ISelf, Size, _IBases...>;

template <class _Interface>
using __finalize = detail::__rebind<_Interface, typename _Interface::__any_t>;

template <class _Interface>
struct __basic_any
    : __finalize<_Interface>
    , private detail::__poly_box<__finalize<_Interface>>
{
  using __box_t = detail::__poly_box<__finalize<_Interface>>;

  _LIBCUDACXX_TEMPLATE(class _Obj)
  _LIBCUDACXX_REQUIRES(_CUDA_VSTD::constructible_from<__box_t, _Obj>)
  __basic_any(_Obj&& __obj) noexcept(noexcept(__box_t(static_cast<_Obj&&>(__obj))))
      : __box_t(static_cast<_Obj&&>(__obj))
  {}
};

using detail::__vcall;

#define _CUDAX_INHERIT_BASIC_ANY_CTOR(_INTERFACE)                                                           \
  _LIBCUDACXX_TEMPLATE(class _Obj)                                                                          \
  _LIBCUDACXX_REQUIRES(_CUDA_VSTD::constructible_from<typename _INTERFACE::__basic_any, _Obj>)              \
  _INTERFACE(_Obj&& __obj) noexcept(noexcept(typename _INTERFACE::__basic_any(static_cast<_Obj&&>(__obj)))) \
      : _INTERFACE::__basic_any(static_cast<_Obj&&>(__obj))                                                 \
  {}                                                                                                        \
  using __dummy = void

struct __any_immovable : __basic_any<detail::__any_immovable<__any_immovable>>
{
  _CUDAX_INHERIT_BASIC_ANY_CTOR(__any_immovable);
};

struct __any_moveonly : __basic_any<detail::__any_moveonly<__any_moveonly>>
{
  _CUDAX_INHERIT_BASIC_ANY_CTOR(__any_moveonly);
};

struct __any_equality_comparable : __basic_any<detail::__any_equality_comparable<__any_equality_comparable>>
{
  _CUDAX_INHERIT_BASIC_ANY_CTOR(__any_equality_comparable);
};

} // namespace cuda::experimental

#endif // __CUDAX_DETAIL_BASIC_ANY_H
