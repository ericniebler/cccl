//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023-24 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TYPE_TRAITS_TYPE_LIST_H
#define _LIBCUDACXX___TYPE_TRAITS_TYPE_LIST_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/type_identity.h>
#include <cuda/std/__type_traits/void_t.h>
#include <cuda/std/__utility/integer_sequence.h>

//! \file type_list.h
//! This file defines a type-list type and some fundamental algorithms on type
//! lists.
//!
//! It also defines a "meta-callable" protocol for parameterizing the type list
//! algorithms, and some higher-order meta-callables. A meta-callable is a class
//! type with a nested \c __call template alias.
//!
//! For the purpose of this file, a "trait type" is a class type with a nested
//! type alias named \c type.

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#ifndef DOXYGEN_SHOULD_SKIP_THIS // Do not document

template <class _Tp>
using __type _LIBCUDACXX_NODEBUG_TYPE = typename _Tp::type;

//! \brief Evaluate a meta-callable with the given arguments
template <class _Fn, class... _Ts>
using __type_call _LIBCUDACXX_NODEBUG_TYPE = typename _Fn::template __call<_Ts...>;

//! \brief Evaluate a unary meta-callable with the given argument
template <class _Fn, class _Ty>
using __type_call1 _LIBCUDACXX_NODEBUG_TYPE = typename _Fn::template __call<_Ty>;

//! \brief Evaluate a binary meta-callable with the given arguments
template <class _Fn, class _Ty, class _Uy>
using __type_call2 _LIBCUDACXX_NODEBUG_TYPE = typename _Fn::template __call<_Ty, _Uy>;

namespace __detail
{
// \brief A helper for determining the arity of a template.
// (This is unused for now, I just want to see if this passes CI.)
template <template <class> class _Fn>
_CCCL_HOST_DEVICE constexpr int __template_arity_aux()
{
  return 1;
}

template <template <class, class> class _Fn>
_CCCL_HOST_DEVICE constexpr int __template_arity_aux()
{
  return 2;
}

template <template <class...> class _Fn, int _Arity = __template_arity_aux<_Fn>()>
_CCCL_HOST_DEVICE constexpr int __template_arity(int)
{
  return _Arity;
}

template <template <class...> class _Fn>
_CCCL_HOST_DEVICE constexpr int __template_arity(long)
{
  return -1;
}

template <class...>
struct _Tn;
template <class>
struct _T1;
template <class, class>
struct _T2;
static_assert(__template_arity<_Tn>(0) == -1, "");
static_assert(__template_arity<_T1>(0) == 1, "");
static_assert(__template_arity<_T2>(0) == 2, "");
} // namespace __detail

//! \brief Turns a class or alias template into a meta-callable
template <template <class...> class _Fn /*, int _Arity = __detail::__template_arity<_Fn>(0)*/>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __type_quote
{
  template <class... _Ts>
  using __call _LIBCUDACXX_NODEBUG_TYPE = _Fn<_Ts...>;
};

//! \brief Turns a unary class or alias template into a meta-callable
template <template <class> class _Fn>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __type_quote1
{
  template <class _Ty>
  using __call _LIBCUDACXX_NODEBUG_TYPE = _Fn<_Ty>;
};

//! \brief Turns a binary class or alias template into a meta-callable
template <template <class, class> class _Fn>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __type_quote2
{
  template <class _Ty, class _Uy>
  using __call _LIBCUDACXX_NODEBUG_TYPE = _Fn<_Ty, _Uy>;
};

//! \brief Turns a trait class template \c _Fn into a meta-callable \c
//! __type_quote_trait such that \c __type_quote_trait<_Fn>::__call<_Ts...> is
//! equivalent to \c _Fn<_Ts...>::type.
template <template <class...> class _Fn>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __type_quote_trait
{
  template <class... _Ts>
  using __call _LIBCUDACXX_NODEBUG_TYPE = __type<_Fn<_Ts...>>;
};

//! \brief Turns a unary trait class template \c _Fn into a meta-callable
//! \c __type_quote_trait1 such that \c __type_quote_trait1<_Fn>::__call<_Ty> is
//! equivalent to \c _Fn<_Ty>::type.
template <template <class> class _Fn>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __type_quote_trait1
{
  template <class _Ty>
  using __call _LIBCUDACXX_NODEBUG_TYPE = __type<_Fn<_Ty>>;
};

//! \brief Turns a binary trait class template \c _Fn into a meta-callable
//! \c __type_quote_trait2 such that \c __type_quote_trait2<_Fn>::__call<_Ty,_Uy>
//! is equivalent to \c _Fn<_Ty,_Uy>::type.
template <template <class, class> class _Fn>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __type_quote_trait2
{
  template <class _Ty, class _Uy>
  using __call _LIBCUDACXX_NODEBUG_TYPE = __type<_Fn<_Ty, _Uy>>;
};

//! \brief A meta-callable that composes two meta-callables
template <class _Fn1, class _Fn2>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __type_compose
{
  template <class... _Ts>
  using __call _LIBCUDACXX_NODEBUG_TYPE = __type_call1<_Fn1, __type_call<_Fn2, _Ts...>>;
};

//! \brief A meta-callable that binds the front arguments to a meta-callable
template <class _Fn, class... _Ts>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __type_bind_front
{
  template <class... _Us>
  using __call _LIBCUDACXX_NODEBUG_TYPE = __type_call<_Fn, _Ts..., _Us...>;
};

//! \brief A meta-callable that binds the back arguments to a meta-callable
template <class _Fn, class... _Ts>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __type_bind_back
{
  template <class... _Us>
  using __call _LIBCUDACXX_NODEBUG_TYPE = __type_call<_Fn, _Us..., _Ts...>;
};

//! \brief A meta-callable that always evaluates to \c _Ty.
template <class _Ty>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __type_always
{
  template <class...>
  using __call _LIBCUDACXX_NODEBUG_TYPE = _Ty;
};

//! \brief A type list
template <class... _Ts>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __type_list
{
  static constexpr size_t const __size = sizeof...(_Ts);

  // A type_list behaves like a meta-callable
  // that takes a meta-callable and applies the
  // elements of the list to it.
  template <class _Fn, class... _Us>
  using __call _LIBCUDACXX_NODEBUG_TYPE = __type_call<_Fn, _Ts..., _Us...>;
};

// Before the addition of inline variables, it was necessary to
// provide a definition for constexpr class static data members.
#  if _CCCL_STD_VER >= 2017 && defined(__cpp_inline_variables) && (__cpp_inline_variables >= 201606L)
template <class... _Ts>
constexpr size_t const __type_list<_Ts...>::__size;
#  endif

//! \brief A pointer to a type list, often used as a function argument.
template <class... _Ts>
using __type_list_ptr = __type_list<_Ts...>*;

//! \brief Return the size of a type list.
template <class _List>
using __type_list_size _LIBCUDACXX_NODEBUG_TYPE = integral_constant<size_t, _List::__size>;

//! \brief Given a type list and a list of types, append the types to the list.
template <class _List, class... _Ts>
using __type_push_back _LIBCUDACXX_NODEBUG_TYPE = __type_call<_List, __type_quote<__type_list>, _Ts...>;

//! \brief Given a type list and a list of types, prepend the types to the list.
template <class _List, class... _Ts>
using __type_push_front _LIBCUDACXX_NODEBUG_TYPE =
  __type_call1<_List, __type_bind_front<__type_quote<__type_list>, _Ts...>>;

namespace __detail
{
template <class _Fn, class _ArgList, class _Enable = void>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __type_callable
{
  using type _LIBCUDACXX_NODEBUG_TYPE = false_type;
};

template <class _Fn, class... _Ts>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __type_callable<_Fn, __type_list<_Ts...>, void_t<__type_call<_Fn, _Ts...>>>
{
  using type _LIBCUDACXX_NODEBUG_TYPE = true_type;
};
} // namespace __detail

//! \brief Test whether a meta-callable is callable with a given set of
//! arguments.
//!
//! Given a meta-callable and a list of arguments, return \c true_type if
//! the meta-callable can be called with the arguments, and \c false_type
//! otherwise.
template <class _Fn, class... _Ts>
using __type_callable _LIBCUDACXX_NODEBUG_TYPE = __type<__detail::__type_callable<_Fn, __type_list<_Ts...>>>;

namespace __detail
{
template <class _Fn, class _ArgList, class _Enable = void>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __type_defer
{};

template <class _Fn, class... _Ts>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __type_defer<_Fn, __type_list<_Ts...>, void_t<__type_call<_Fn, _Ts...>>>
{
  using type _LIBCUDACXX_NODEBUG_TYPE = __type_call<_Fn, _Ts...>;
};
} // namespace __detail

//! \brief Defer the evaluation of a meta-callable with a list of arguments.
//!
//! Given a meta-callable and a list of arguments, return a trait type \c T
//! where \c T::type is the result of evaluating the meta-callable with the
//! arguments, or if the meta-callable is not callable with the arguments, a
//! class type without a nested \c ::type type alias.
template <class _Fn, class... _Ts>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __type_defer : __detail::__type_defer<_Fn, __type_list<_Ts...>>
{};

// Implementation for indexing into a list of types:
#  if defined(__cpp_pack_indexing)

template <size_t _Ip, class... _Ts>
using __type_index_c _LIBCUDACXX_NODEBUG_TYPE = _Ts...[_Ip];

template <class _Ip, class... _Ts>
using __type_index _LIBCUDACXX_NODEBUG_TYPE = _Ts...[_Ip::value];

// Versions of nvcc prior to 12.0 have trouble with pack expansion into
// __type_pack_element in an alias template, so we use the fall-back
// implementation instead.
#  elif __has_builtin(__type_pack_element) && (!defined(_CCCL_CUDA_COMPILER_NVCC) || _CCCL_CUDACC_VER_MAJOR > 11)

namespace __detail
{
// On some versions of gcc, __type_pack_element cannot be mangled so
// hide it behind a named template.
template <size_t _Ip>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __type_index_fn
{
  template <class... _Ts>
  using __call _LIBCUDACXX_NODEBUG_TYPE = __type_pack_element<_Ip, _Ts...>;
};
} // namespace __detail

template <size_t _Ip, class... _Ts>
using __type_index_c _LIBCUDACXX_NODEBUG_TYPE = __type_call<__detail::__type_index_fn<_Ip>, _Ts...>;

template <class _Ip, class... _Ts>
using __type_index _LIBCUDACXX_NODEBUG_TYPE = __type_call<__detail::__type_index_fn<_Ip::value>, _Ts...>;

#  else // ^^^ __has_builtin(__type_pack_element) ^^^ / vvv !__has_builtin(__type_pack_element) vvv

namespace __detail
{
template <size_t>
using __void_ptr _LIBCUDACXX_NODEBUG_TYPE = void*;

template <class _Ty>
using __type_ptr _LIBCUDACXX_NODEBUG_TYPE = __type_identity<_Ty>*;

template <class _Is>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __type_index_fn_;

template <size_t... _Is>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __type_index_fn_<index_sequence<_Is...>>
{
  template <class _Up>
  _CCCL_HOST_DEVICE static _Up __call_(__void_ptr<_Is>..., _Up*, ...);

  template <class... _Ts>
  using __call _LIBCUDACXX_NODEBUG_TYPE = __type<decltype(__type_index_fn::__call_(__type_ptr<_Ts>{nullptr}...))>;
};

template <size_t _Ip>
using __type_index_fn _LIBCUDACXX_NODEBUG_TYPE = __type_index_fn_<make_index_sequence<_Ip>>;
} // namespace __detail

template <size_t _Ip, class... _Ts>
using __type_index_c _LIBCUDACXX_NODEBUG_TYPE = __type_call<__detail::__type_index_fn<_Ip>, _Ts...>;

template <class _Ip, class... _Ts>
using __type_index _LIBCUDACXX_NODEBUG_TYPE = __type_call<__detail::__type_index_fn<_Ip::value>, _Ts...>;

#  endif // !__has_builtin(__type_pack_element)

namespace __detail
{
template <size_t _Ip>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __type_at_fn
{
  template <class... _Ts>
  using __call _LIBCUDACXX_NODEBUG_TYPE = __type_index_c<_Ip, _Ts...>;
};
} // namespace __detail

//! \brief Given a type list and an index, return the type at that index.
template <size_t _Ip, class _List>
using __type_at_c _LIBCUDACXX_NODEBUG_TYPE = __type_call1<_List, __detail::__type_at_fn<_Ip>>;

//! \brief Given a type list and an index, return the type at that index.
template <class _Ip, class _List>
using __type_at _LIBCUDACXX_NODEBUG_TYPE = __type_call1<_List, __detail::__type_at_fn<_Ip::value>>;

//! \brief Given a type list return the type at the front of the list.
template <class _List>
using __type_front _LIBCUDACXX_NODEBUG_TYPE = __type_at_c<0, _List>;

//! \brief Given a type list return the type at the back of the list.
template <class _List>
using __type_back _LIBCUDACXX_NODEBUG_TYPE = __type_at_c<_List::__size - 1, _List>;

namespace __detail
{
template <size_t _Count>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __type_maybe_concat_fn
{
  using __next_t = __type_maybe_concat_fn<(_Count < 8 ? 0 : _Count - 8)>;

  template <class... _Ts,
            class... _As,
            class... _Bs,
            class... _Cs,
            class... _Ds,
            class... _Es,
            class... _Fs,
            class... _Gs,
            class... _Hs,
            class... _Tail>
  _CCCL_HOST_DEVICE static auto __fn(
    __type_list_ptr<_Ts...>, // state
    __type_list_ptr<_As...>, // 1
    __type_list_ptr<_Bs...>, // 2
    __type_list_ptr<_Cs...>, // 3
    __type_list_ptr<_Ds...>, // 4
    __type_list_ptr<_Es...>, // 5
    __type_list_ptr<_Fs...>, // 6
    __type_list_ptr<_Gs...>, // 7
    __type_list_ptr<_Hs...>, // 8
    _Tail*... __tail) // rest
    -> decltype(__next_t::__fn(
      __type_list_ptr<_Ts..., _As..., _Bs..., _Cs..., _Ds..., _Es..., _Fs..., _Gs..., _Hs...>{nullptr},
      __tail...,
      __type_list_ptr<>{nullptr},
      __type_list_ptr<>{nullptr},
      __type_list_ptr<>{nullptr},
      __type_list_ptr<>{nullptr},
      __type_list_ptr<>{nullptr},
      __type_list_ptr<>{nullptr},
      __type_list_ptr<>{nullptr}));
};

template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __type_maybe_concat_fn<0>
{
  template <class... _Ts>
  _CCCL_HOST_DEVICE static auto __fn(__type_list_ptr<_Ts...>, ...) -> __type_list<_Ts...>;
};

struct _CCCL_TYPE_VISIBILITY_DEFAULT __type_concat_fn
{
  template <class... _Lists>
  using __call = decltype(__type_maybe_concat_fn<sizeof...(_Lists)>::__fn(
    __type_list_ptr<>{nullptr},
    static_cast<_Lists*>(nullptr)...,
    __type_list_ptr<>{nullptr},
    __type_list_ptr<>{nullptr},
    __type_list_ptr<>{nullptr},
    __type_list_ptr<>{nullptr},
    __type_list_ptr<>{nullptr},
    __type_list_ptr<>{nullptr},
    __type_list_ptr<>{nullptr}));
};
} // namespace __detail

//! \brief Concatenate a list of type lists into a single type list.
//!
//! When passed no type lists, \c __type_concat returns an empty type list.
template <class... _Lists>
using __type_concat _LIBCUDACXX_NODEBUG_TYPE = __type_call<__detail::__type_concat_fn, _Lists...>;

//! \brief Given a list of type lists, concatenate all the lists into one.
//!
//! When passed an empty type list, \c __type_flatten returns an empty type list.
template <class _ListOfLists>
using __type_flatten _LIBCUDACXX_NODEBUG_TYPE = __type_call1<_ListOfLists, __type_quote<__type_concat>>;

namespace __detail
{
template <bool _IsEmpty>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __type_maybe_find_if_fn // Type list is not empty
{
  template <class _Fn, class _Head, class... _Tail>
  using __call _LIBCUDACXX_NODEBUG_TYPE =
    _If<__type_call1<_Fn, _Head>::value,
        __type_list<_Head, _Tail...>,
        __type_call<__type_maybe_find_if_fn<sizeof...(_Tail) == 0>, _Fn, _Tail...>>;
};

template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __type_maybe_find_if_fn<true> // Type list is empty
{
  template <class, class... _None>
  using __call _LIBCUDACXX_NODEBUG_TYPE = __type_list<>;
};

template <class _Fn>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __type_find_if_fn
{
  template <class... _Ts>
  using __call _LIBCUDACXX_NODEBUG_TYPE = __type_call<__type_maybe_find_if_fn<sizeof...(_Ts) == 0>, _Fn, _Ts...>;
};
} // namespace __detail

//! \brief Given a type list and a predicate, find the first type in a list that
//! satisfies a predicate. It returns a type list containing the first type that
//! satisfies the predicate and all the types after it.
//!
//! If no type in the list satisfies the predicate, \c __type_find_if
//! returns an empty type list.
template <class _List, class _Fn>
using __type_find_if _LIBCUDACXX_NODEBUG_TYPE = __type_call1<_List, __detail::__type_find_if_fn<_Fn>>;

namespace __detail
{
template <class _Fn>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __type_transform_fn
{
  template <class... _Ts>
  using __call _LIBCUDACXX_NODEBUG_TYPE = __type_list<__type_call1<_Fn, _Ts>...>;
};
} // namespace __detail

//! \brief Given a type list and a unary meta-callable, apply the meta-callable
//! to each type in the list. It returns a new type list containing the results.
template <class _List, class _Fn>
using __type_transform _LIBCUDACXX_NODEBUG_TYPE = __type_call1<_List, __detail::__type_transform_fn<_Fn>>;

//
// Implementation for folding a type list either left or right
//
namespace __detail
{
template <bool>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __type_maybe_fold_right_fn
{
  template <class _Fn, class _State, class _Head, class... _Tail>
  using __call _LIBCUDACXX_NODEBUG_TYPE =
    __type_call<__type_maybe_fold_right_fn<sizeof...(_Tail) == 0>, _Fn, __type_call2<_Fn, _State, _Head>, _Tail...>;
};

template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __type_maybe_fold_right_fn<true> // empty pack
{
  template <class _Fn, class _State, class...>
  using __call _LIBCUDACXX_NODEBUG_TYPE = _State;
};

template <class _Init, class _Fn>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __type_fold_right_fn
{
  template <class... _Ts>
  using __call _LIBCUDACXX_NODEBUG_TYPE =
    __type_call<__type_maybe_fold_right_fn<sizeof...(_Ts) == 0>, _Fn, _Init, _Ts...>;
};

template <bool>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __type_maybe_fold_left_fn
{
  template <class _Fn, class _State, class _Head, class... _Tail>
  using __call _LIBCUDACXX_NODEBUG_TYPE =
    __type_call2<_Fn, __type_call<__type_maybe_fold_left_fn<sizeof...(_Tail) == 0>, _Fn, _State, _Tail...>, _Head>;
};

template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __type_maybe_fold_left_fn<true> // empty pack
{
  template <class _Fn, class _State, class...>
  using __call _LIBCUDACXX_NODEBUG_TYPE = _State;
};

template <class _Init, class _Fn>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __type_fold_left_fn
{
  template <class... _Ts>
  using __call _LIBCUDACXX_NODEBUG_TYPE =
    __type_call<__type_maybe_fold_left_fn<sizeof...(_Ts) == 0>, _Fn, _Init, _Ts...>;
};

} // namespace __detail

//! \brief Fold a type list from the right with a binary meta-callable and an
//! initial state.
template <class _List, class _Init, class _Fn>
using __type_fold_right _LIBCUDACXX_NODEBUG_TYPE = __type_call1<_List, __detail::__type_fold_right_fn<_Init, _Fn>>;

//! \brief Fold a type list from the left with a binary meta-callable and an
//! initial state.
template <class _List, class _Init, class _Fn>
using __type_fold_left _LIBCUDACXX_NODEBUG_TYPE = __type_call1<_List, __detail::__type_fold_left_fn<_Init, _Fn>>;

namespace __detail
{
template <class _Ty>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __type_remove_fn
{
  template <class _Uy>
  using __call _LIBCUDACXX_NODEBUG_TYPE = _If<_CCCL_TRAIT(is_same, _Ty, _Uy), __type_list<>, __type_list<_Uy>>;
};
} // namespace __detail

//! \brief Remove all occurances of a type from a type list
template <class _List, class _Ty>
using __type_remove _LIBCUDACXX_NODEBUG_TYPE = __type_flatten<__type_transform<_List, __detail::__type_remove_fn<_Ty>>>;

namespace __detail
{
// _State is the list of type lists being built by the __type_fold_left
template <class _State, class _List>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __type_cartesian_product_fn
{
  template <class _Ty>
  struct __lambda0
  {
    template <class _List2>
    using __lambda1 _LIBCUDACXX_NODEBUG_TYPE = __type_list<__type_push_front<_List2, _Ty>>;

    using type _LIBCUDACXX_NODEBUG_TYPE = __type_flatten<__type_transform<_State, __type_quote1<__lambda1>>>;
  };

  using type _LIBCUDACXX_NODEBUG_TYPE = __type_flatten<__type_transform<_List, __type_quote_trait1<__lambda0>>>;
};
} // namespace __detail
/// \endcond

//! \brief Given a list of lists \c _Lists, return a new list of lists that is
//! the cartesian product.
//!
//! \par Complexity
//! `O(N * M)`, where `N` is the size of the outer list, and
//! `M` is the size of the inner lists.
template <class... _Lists>
using __type_cartesian_product _LIBCUDACXX_NODEBUG_TYPE =
  __type_fold_left<__type_list<_Lists...>,
                   __type_list<__type_list<>>,
                   __type_quote_trait2<__detail::__type_cartesian_product_fn>>;

// A unary meta-callable for converting a type to its size in bytes
struct _CCCL_TYPE_VISIBILITY_DEFAULT __type_sizeof
{
  template <class _Ty>
  using __call _LIBCUDACXX_NODEBUG_TYPE = integral_constant<size_t, sizeof(_Ty)>;
};

//! \brief Perform a logical AND operation on a list of Boolean types.
//!
//! \note The AND operation is not short-circuiting.
struct _CCCL_TYPE_VISIBILITY_DEFAULT __type_strict_and
{
#  ifndef _LIBCUDACXX_HAS_NO_FOLD_EXPRESSIONS
  template <class... _Ts>
  using __call _LIBCUDACXX_NODEBUG_TYPE = bool_constant<(_Ts::value && ...)>;
#  else
  template <class... _Ts>
  using __call _LIBCUDACXX_NODEBUG_TYPE = bool_constant<_CCCL_TRAIT(
    is_same, integer_sequence<bool, true, _Ts::value...>, integer_sequence<bool, _Ts::value..., true>)>;
#  endif
};

//! \brief Perform a logical OR operation on a list of Boolean types.
//!
//! \note The OR operation is not short-circuiting.
struct _CCCL_TYPE_VISIBILITY_DEFAULT __type_strict_or
{
#  ifndef _LIBCUDACXX_HAS_NO_FOLD_EXPRESSIONS
  template <class... _Ts>
  using __call _LIBCUDACXX_NODEBUG_TYPE = bool_constant<(_Ts::value || ...)>;
#  else
  template <class... _Ts>
  using __call _LIBCUDACXX_NODEBUG_TYPE = bool_constant<!_CCCL_TRAIT(
    is_same, integer_sequence<bool, false, _Ts::value...>, integer_sequence<bool, _Ts::value..., false>)>;
#  endif
};

//! \brief Perform a logical NOT operation on a Boolean type.
//!
//! \note The AND operation is not short-circuiting.
struct _CCCL_TYPE_VISIBILITY_DEFAULT __type_not
{
  template <class _Ty>
  using __call _LIBCUDACXX_NODEBUG_TYPE = bool_constant<(!_Ty::value)>;
};

//! \brief Test whether two integral constants are equal.
struct _CCCL_TYPE_VISIBILITY_DEFAULT __type_equal
{
  template <class _Ty, class _Uy>
  using __call _LIBCUDACXX_NODEBUG_TYPE = bool_constant<(_Ty::value == _Uy::value)>;
};

//! \brief Test whether two integral constants are not equal.
struct _CCCL_TYPE_VISIBILITY_DEFAULT __type_not_equal
{
  template <class _Ty, class _Uy>
  using __call _LIBCUDACXX_NODEBUG_TYPE = bool_constant<(_Ty::value != _Uy::value)>;
};

//! \brief Test whether one integral constant is less than another.
struct _CCCL_TYPE_VISIBILITY_DEFAULT __type_less
{
  template <class _Ty, class _Uy>
  using __call _LIBCUDACXX_NODEBUG_TYPE = bool_constant<(_Ty::value < _Uy::value)>;
};

//! \brief Test whether one integral constant is less than or equal to another.
struct _CCCL_TYPE_VISIBILITY_DEFAULT __type_less_equal
{
  template <class _Ty, class _Uy>
  using __call _LIBCUDACXX_NODEBUG_TYPE = bool_constant<(_Ty::value <= _Uy::value)>;
};

//! \brief Test whether one integral constant is greater than another.
struct _CCCL_TYPE_VISIBILITY_DEFAULT __type_greater
{
  template <class _Ty, class _Uy>
  using __call _LIBCUDACXX_NODEBUG_TYPE = bool_constant<(_Ty::value > _Uy::value)>;
};

//! \brief Test whether one integral constant is greater than or equal to another.
struct _CCCL_TYPE_VISIBILITY_DEFAULT __type_greater_equal
{
  template <class _Ty, class _Uy>
  using __call _LIBCUDACXX_NODEBUG_TYPE = bool_constant<(_Ty::value >= _Uy::value)>;
};

//! \brief A pair of types
template <class _First, class _Second>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __type_pair
{
  using __first _LIBCUDACXX_NODEBUG_TYPE  = _First;
  using __second _LIBCUDACXX_NODEBUG_TYPE = _Second;
};

//! \brief Retreive the first of a pair of types
//! \pre \c _Pair is a specialization of \c __type_pair
template <class _Pair>
using __type_pair_first _LIBCUDACXX_NODEBUG_TYPE = typename _Pair::__first;

//! \brief Retreive the second of a pair of types
//! \pre \c _Pair is a specialization of \c __type_pair
template <class _Pair>
using __type_pair_second _LIBCUDACXX_NODEBUG_TYPE = typename _Pair::__second;

//! \brief A list of compile-time values, and a meta-callable that accepts a
//! meta-callable and evaluates it with the values, each value wrapped in an
//! integral constant wrapper.
template <class _Ty, _Ty... _Values>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __type_value_list : __type_list<integral_constant<_Ty, _Values>...>
{
  using __type _LIBCUDACXX_NODEBUG_TYPE = _Ty;
};

namespace __detail
{
template <class _Ty, _Ty _Start, _Ty _Stride, _Ty... _Is>
_CCCL_HOST_DEVICE auto
__type_iota_fn(integer_sequence<_Ty, _Is...>*) -> __type_value_list<_Ty, _Ty(_Start + (_Is * _Stride))...>;
} // namespace __detail

//! \brief Return an \c __type_value_list of size \c _Size starting at \c _Start
//! and incrementing by \c _Stride.
template <class _Ty, _Ty _Start, _Ty _Size, _Ty _Stride = _Ty(1)>
using __type_iota _LIBCUDACXX_NODEBUG_TYPE =
  decltype(__detail::__type_iota_fn<_Ty, _Start, _Stride>(static_cast<make_integer_sequence<_Ty, _Size>*>(nullptr)));

#endif // DOXYGEN_SHOULD_SKIP_THIS

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___TYPE_TRAITS_TYPE_LIST_H
