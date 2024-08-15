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
#include <cuda/std/__type_traits/decay.h>
#include <cuda/std/__type_traits/is_base_of.h>
#include <cuda/std/__type_traits/is_valid_expansion.h>
#include <cuda/std/__utility/declval.h>
#include <cuda/std/__utility/exchange.h>

#include <new> // IWYU pragma: keep (needed for placement new)
#include <typeinfo>

namespace cuda::experimental
{
namespace detail
{
template <class _Ty, class _Uy>
_LIBCUDACXX_CONCEPT __not_derived_from = !_CUDA_VSTD::is_base_of_v<_Uy, _CUDA_VSTD::decay_t<_Ty>>;

template <class _Interface, class _Ty = _Interface>
using __members_of = typename _Interface::template members<_Ty>;

template <class _Ty, class _Interface>
_LIBCUDACXX_CONCEPT __satisfies = _CUDA_VSTD::_IsValidExpansion<__members_of, _Interface, _Ty>::value;

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
_CCCL_GLOBAL_CONSTANT __fn_t<_Ret, void*, _As...>* __vfn<_Ty, _Mbr, _Tiny, _Ret (_Cy::*)(_As...)> =
  +[](void* __pv, _As... __as) -> _Ret { // __pv is a pointer to a __storage object.
  _Cy* __obj = _Tiny ? static_cast<_Ty*>(__pv) : *static_cast<_Ty**>(__pv);
  return (__obj->*_Mbr)(static_cast<_As&&>(__as)...);
};

template <class _Ty, auto _Mbr, bool _Tiny, class _Ret, class _Cy, class... _As>
_CCCL_GLOBAL_CONSTANT __fn_t<_Ret, void const*, _As...>* __vfn<_Ty, _Mbr, _Tiny, _Ret (_Cy::*)(_As...) const> =
  +[](void const* __pv, _As... __as) -> _Ret { // __pv is a pointer to a __storage object.
  _Cy const* __obj = _Tiny ? static_cast<_Ty const*>(__pv) : *static_cast<_Ty const* const*>(__pv);
  return (__obj->*_Mbr)(static_cast<_As&&>(__as)...);
};

template <class _Ty, auto _Mbr, bool _Tiny, class _Ret, class _Cy, class... _As>
_CCCL_GLOBAL_CONSTANT __fn_t<_Ret, void*, _As...>* __vfn<_Ty, _Mbr, _Tiny, _Ret (*)(_Cy&, _As...)> =
  +[](void* __pv, _As... __as) -> _Ret { // __pv is a pointer to a __storage object.
  _Cy* __obj = _Tiny ? static_cast<_Ty*>(__pv) : *static_cast<_Ty**>(__pv);
  return _Mbr(*__obj, static_cast<_As&&>(__as)...);
};

template <class _Ty, auto _Mbr, bool _Tiny, class _Ret, class _Cy, class... _As>
_CCCL_GLOBAL_CONSTANT __fn_t<_Ret, void const*, _As...>* __vfn<_Ty, _Mbr, _Tiny, _Ret (*)(_Cy const&, _As...)> =
  +[](void const* __pv, _As... __as) -> _Ret { // __pv is a pointer to a __storage object.
  _Cy const* __obj = _Tiny ? static_cast<_Ty const*>(__pv) : *static_cast<_Ty const* const*>(__pv);
  return _Mbr(*__obj, static_cast<_As&&>(__as)...);
};

template <class _Mbrs>
using __vtable_of = typename _Mbrs::__vtable;

template <class _Interface>
using __base_vtable_of = typename _Interface::__base::__vtable;

template <class _Ty, auto _Mbr>
struct __member
{
  using __fn_t = decltype(__vfn<_Ty, _Mbr, true>);
  __fn_t __fn_;
};

template <class _Ty, auto... _Mbrs>
struct __members
{
  struct __vtable : __member<_Ty, _Mbrs>...
  {};

  template <class _VTable, bool _Tiny>
  static constexpr _VTable __mk_vtable() noexcept
  {
    return _VTable{{__vfn<_Ty, _Mbrs, _Tiny>}...};
  }
};

template <class _Interface>
struct __extensible;

template <template <class...> class _Interface, class _Base, class... _Ts>
struct __extensible<_Interface<_Base, _Ts...>>
{
  using __base_t = _Base;

  template <class _Derived>
  using __rebind = _Interface<_Derived, _Ts...>;
};

template <class _Interface, class _Derived>
using __rebind = typename _Interface::template __rebind<_Derived>;

template <class... _Interfaces>
struct __extends
{
  template <class _Derived>
  struct __base : __rebind<_Interfaces, _Derived>...
  {
    struct __vtable : __vtable_of<__members_of<__rebind<_Interfaces, _Derived>, _Derived>>...
    {};

    template <class _Ty, bool _Tiny>
    static constexpr __vtable __mk_vtable() noexcept
    {
      return __vtable{
        {__members_of<__rebind<_Interfaces, _Derived>, _Ty>::
           template __mk_vtable<__vtable_of<__members_of<__rebind<_Interfaces, _Derived>, _Derived>>, _Tiny>()}...};
    }
  };
};

template <class _Interface>
struct __vtable_for
    : __base_vtable_of<_Interface> // for base interfaces
    , __vtable_of<__members_of<_Interface>>
{
  using __interface_t = _Interface;

  ::std::type_info const* __type_;
  bool __tiny_;
  void (*__destroy_)(void*) noexcept;
  void (*__move_)(void*, void*) noexcept;
  void (*__copy_)(void*, void const*);

  template <class _Ty>
  _Ty& __as(void* __pv) const noexcept
  {
    _LIBCUDACXX_ASSERT(typeid(_Ty) == *__type_, "");
    return __tiny_ ? *static_cast<_Ty*>(__pv) : **static_cast<_Ty**>(__pv);
  }

  template <class _Ty>
  _Ty const& __as(void const* __pv) const noexcept
  {
    _LIBCUDACXX_ASSERT(typeid(_Ty) == *__type_, "");
    return __tiny_ ? *static_cast<_Ty const*>(__pv) : **static_cast<_Ty const* const*>(__pv);
  }
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

template <class _Interface, std::size_t _Size>
struct __basic_any_impl
{
  using __interface_t = _Interface;

  _LIBCUDACXX_TEMPLATE(class _Obj, class _Ty = _CUDA_VSTD::decay_t<_Obj>)
  _LIBCUDACXX_REQUIRES(__not_derived_from<_Obj, __basic_any_impl> _LIBCUDACXX_AND __satisfies<_Ty, _Interface>)
  __basic_any_impl(_Obj&& __obj) //
    noexcept(noexcept(_Ty(_CUDA_VSTD::declval<_Obj>())) && (sizeof(_Ty) <= _Size))
      : __vptr_(__vptr_for<_Ty>())
  {
    if constexpr (sizeof(_Ty) <= _Size)
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
  __basic_any_impl(__basic_any_impl&& __other) noexcept
      : __vptr_(_CUDA_VSTD::exchange(__other.__vptr_, nullptr))
  {
    if (__vptr_)
    {
      __vptr()->__move_(&__obj_, &__other.__obj_);
    }
  }

  _LIBCUDACXX_TEMPLATE(class _Cy = _Interface)
  _LIBCUDACXX_REQUIRES(_CUDA_VSTD::copy_constructible<_Cy>)
  __basic_any_impl(__basic_any_impl const& __other)
      : __vptr_(__other.__vptr_)
  {
    if (__vptr_)
    {
      __vptr()->__copy_(&__obj_, &__other.__obj_);
    }
  }

  ~__basic_any_impl()
  {
    if (__vptr_)
    {
      __vptr()->__destroy_(&__obj_);
    }
  }

  _LIBCUDACXX_TEMPLATE(class _Cy = _Interface)
  _LIBCUDACXX_REQUIRES(_CUDA_VSTD::movable<_Cy>)
  [[maybe_unused]]
  __basic_any_impl& operator=(__basic_any_impl&& __other) noexcept
  {
    if (this != &__other)
    {
      if (__vptr_)
      {
        __vptr()->__destroy_(&__obj_);
        __vptr_ = nullptr;
      }

      if (__other.__vptr_)
      {
        __vptr_ = _CUDA_VSTD::exchange(__other.__vptr_, nullptr);
        __vptr()->__move_(&__obj_, &__other.__obj_);
      }
    }

    return *this;
  }

  _LIBCUDACXX_TEMPLATE(class _Cy = _Interface)
  _LIBCUDACXX_REQUIRES(_CUDA_VSTD::copyable<_Cy>)
  [[maybe_unused]]
  __basic_any_impl& operator=(__basic_any_impl const& __other)
  {
    return this == &__other ? *this : operator=(__basic_any_impl(__other));
  }

  template <class _Ty>
  static void const* __vptr_for() noexcept
  {
    using __vtable_t      = __vtable_for<_Interface>;
    using __vtable_base_t = __vtable_of<__members_of<_Interface>>;
    constexpr bool __tiny = (sizeof(_Ty) <= _Size);

    static constexpr __vtable_t __s_vtable{
      _Interface::template __mk_vtable<_Ty, __tiny>(),
      __members_of<_Interface, _Ty>::template __mk_vtable<__vtable_base_t, __tiny>(),
      &typeid(_Ty),
      __tiny,
      &__destroy_<_Ty, __tiny>,
      &__move_<_Ty, __tiny>,
      &__copy_<_Ty, __tiny>};

    return &__s_vtable;
  }

  auto __vptr() const noexcept
  {
    return static_cast<__vtable_for<_Interface> const*>(__vptr_);
  }

  void const* __vptr_{};
  __storage<_Size> __obj_{};
};

_CCCL_GLOBAL_CONSTANT std::size_t __default_small_buffer_size = 3 * sizeof(void*);

constexpr std::size_t __buffer_size(std::size_t __size) noexcept
{
  return 0 == __size ? __default_small_buffer_size : (__size < sizeof(void*) ? sizeof(void*) : __size);
}

template <class _Self, class _Base>
auto __basic_any_cast(_Base& __base) noexcept -> typename _Self::__basic_any_t&
{
  static_assert(_CUDA_VSTD::same_as<_Self, typename _Base::__base_t>);
  using __basic_any_t = typename _Self::__basic_any_t;
  static_assert(_CUDA_VSTD::is_base_of_v<__basic_any_t, _Self>);
  // C-style cast to skip accessibility check:
  return (__basic_any_t&) static_cast<_Self&>(__base);
}

template <class _Self, class _Base>
auto __basic_any_cast(_Base const& __base) noexcept -> typename _Self::__basic_any_t const&
{
  static_assert(_CUDA_VSTD::same_as<_Self, typename _Base::__base_t>);
  using __basic_any_t = typename _Self::__basic_any_t;
  static_assert(_CUDA_VSTD::is_base_of_v<__basic_any_t, _Self>);
  // C-style cast to skip accessibility check:
  return (__basic_any_t const&) static_cast<_Self const&>(__base);
}

template <auto _Mbr, class _Ret, class _Cy, class... _As>
_Ret __vcall_impl(_Cy& __self, _As... __as)
{
  using __self_t                       = typename _Cy::__base_t;
  auto& __base                         = detail::__basic_any_cast<__self_t>(__self);
  const __member<__self_t, _Mbr> __mbr = *__base.__vptr();
  return __mbr.__fn_(&__base.__obj_, static_cast<_As&&>(__as)...);
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

template <class _Self>
struct __any_immovable : __extensible<__any_immovable<_Self>>
{
  __any_immovable()                                  = default;
  __any_immovable(__any_immovable&&)                 = delete;
  __any_immovable(__any_immovable const&)            = delete;
  __any_immovable& operator=(__any_immovable&&)      = delete;
  __any_immovable& operator=(__any_immovable const&) = delete;

  template <class _Ty>
  using members = __members<_Ty>;
};

template <class _Self>
struct __any_moveonly : __extensible<__any_moveonly<_Self>>
{
  __any_moveonly()                                 = default;
  __any_moveonly(__any_moveonly&&)                 = default;
  __any_moveonly(__any_moveonly const&)            = delete;
  __any_moveonly& operator=(__any_moveonly&&)      = default;
  __any_moveonly& operator=(__any_moveonly const&) = delete;

  template <class _Ty>
  using members = __members<_Ty>;
};

template <class _Self>
struct __any_equality_comparable : __extensible<__any_equality_comparable<_Self>>
{
  template <class _Ty>
  static auto __equal(_Ty const& __lhs, _Self const& __rhs) -> decltype(__lhs == __lhs)
  {
    auto& __rhs_base = detail::__basic_any_cast<_Self>(__rhs);
    return __lhs == __rhs_base.__vptr()->template __as<_Ty>(&__rhs_base.__obj_);
  }

  friend bool operator==(const _Self& __lhs, const _Self& __rhs)
  {
    auto& __lhs_base = detail::__basic_any_cast<_Self>(__lhs);
    auto& __rhs_base = detail::__basic_any_cast<_Self>(__rhs);

    _LIBCUDACXX_ASSERT(__lhs_base.__vptr_ && __rhs_base.__vptr_, "comparison to moved-from object");

    // Check that the two objects contain the same type:
    if (*__lhs_base.__vptr()->__type_ != *__rhs_base.__vptr()->__type_)
    {
      return false;
    }

    return __vcall<&__equal<_Self>>(__lhs, __rhs);
  }

  friend bool operator!=(const _Self& __lhs, const _Self& __rhs)
  {
    return !(__lhs == __rhs);
  }

  template <class _Ty>
  using members = __members<_Ty, &__equal<_Ty>>;
};

} // namespace detail

template <class _Interface, auto... _Mbrs>
using __members = detail::__members<_Interface, _Mbrs...>;

template <class _Interface>
using __extensible = detail::__extensible<_Interface>;

template <class _Interface, std::size_t _Size = 0, class... _BaseInterfaces>
struct __basic_any
    : private detail::__basic_any_impl<_Interface, detail::__buffer_size(_Size)>
    , detail::__extends<_BaseInterfaces...>::template __base<_Interface>
{
  using __basic_any_t = detail::__basic_any_impl<_Interface, detail::__buffer_size(_Size)>;

  _LIBCUDACXX_TEMPLATE(class _Obj)
  _LIBCUDACXX_REQUIRES(_CUDA_VSTD::constructible_from<__basic_any_t, _Obj>)
  __basic_any(_Obj&& __obj) noexcept(noexcept(__basic_any_t(static_cast<_Obj&&>(__obj))))
      : __basic_any_t(static_cast<_Obj&&>(__obj))
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

struct __any_immovable
    : detail::__any_immovable<__any_immovable>
    , __basic_any<__any_immovable>
{
  _CUDAX_INHERIT_BASIC_ANY_CTOR(__any_immovable);
};

struct __any_moveonly
    : detail::__any_moveonly<__any_moveonly>
    , __basic_any<__any_moveonly>
{
  _CUDAX_INHERIT_BASIC_ANY_CTOR(__any_moveonly);
};

struct __any_equality_comparable
    : detail::__any_equality_comparable<__any_equality_comparable>
    , __basic_any<__any_equality_comparable>
{
  _CUDAX_INHERIT_BASIC_ANY_CTOR(__any_equality_comparable);
};

} // namespace cuda::experimental

#endif // __CUDAX_DETAIL_BASIC_ANY_H
