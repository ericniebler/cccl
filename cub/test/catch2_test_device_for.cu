/******************************************************************************
 * Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_for.cuh>

#include <thrust/count.h>
#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/equal.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/sequence.h>

#include "catch2_test_launch_helper.h"
#include <c2h/catch2_test_helper.h>

// %PARAM% TEST_LAUNCH lid 0:1:2

DECLARE_LAUNCH_WRAPPER(cub::DeviceFor::ForEach, device_for_each);
DECLARE_LAUNCH_WRAPPER(cub::DeviceFor::ForEachN, device_for_each_n);

struct incrementer_t
{
  int* d_counts;

  template <class OffsetT>
  __device__ void operator()(OffsetT i)
  {
    atomicAdd(d_counts + i, 1); // Check if `i` was served more than once
  }
};

template <class OffsetT>
class offset_proxy_t
{
  OffsetT m_offset;

public:
  __host__ __device__ offset_proxy_t(OffsetT offset)
      : m_offset(offset)
  {}

  __host__ __device__ operator OffsetT() const
  {
    return m_offset;
  }
};

struct referencing_operator_t
{
  const std::size_t* d_input;
  const std::size_t magic_value;

  __device__ void operator()(const std::size_t& i) const
  {
    if (i == magic_value)
    {
      const std::size_t* d_ptr    = &i;
      const auto offset           = static_cast<std::size_t>(d_ptr - d_input);
      const_cast<std::size_t&>(i) = offset;
    }
  }
};

C2H_TEST("Device for each works", "[for][device]")
{
  constexpr int max_items = 5000000;
  constexpr int min_items = 1;

  using offset_t = int;

  const offset_t num_items = GENERATE_COPY(
    take(3, random(min_items, max_items)),
    values({
      min_items,
      max_items,
    }));

  c2h::device_vector<offset_proxy_t<offset_t>> input(num_items, offset_t{});
  thrust::sequence(c2h::device_policy, input.begin(), input.end(), offset_t{});
  c2h::device_vector<int> counts(num_items);
  int* d_counts = thrust::raw_pointer_cast(counts.data());

  device_for_each(input.begin(), input.end(), incrementer_t{d_counts});

  const auto num_of_once_marked_items =
    static_cast<offset_t>(thrust::count(c2h::device_policy, counts.begin(), counts.end(), 1));

  REQUIRE(num_of_once_marked_items == num_items);
}

C2H_TEST("Device for each works with bad operators", "[for][device]")
{
  constexpr int max_items = 5000000;
  constexpr int min_items = 1;

  const std::size_t num_items = GENERATE_COPY(
    take(3, random(min_items, max_items)),
    values({
      min_items,
      max_items,
    }));

  const std::size_t magic_value = num_items + 1;
  c2h::device_vector<std::size_t> input(num_items, magic_value);
  const std::size_t* d_input = thrust::raw_pointer_cast(input.data());

  device_for_each(input.begin(), input.end(), referencing_operator_t{d_input, magic_value});

  REQUIRE(thrust::equal(c2h::device_policy, input.begin(), input.end(), thrust::make_counting_iterator(std::size_t{})));
}

C2H_TEST("Device for each works with unaligned vectors", "[for][device]")
{
  constexpr int max_items = 5000000;
  constexpr int min_items = 1;

  const int num_items = GENERATE_COPY(
    take(3, random(min_items, max_items)),
    values({
      min_items,
      max_items,
    }));

  const int offset = GENERATE(1, 2, 3);

  c2h::device_vector<int> counts(num_items);
  c2h::device_vector<int> input(num_items + offset);
  thrust::sequence(c2h::device_policy, input.begin() + offset, input.end());

  int* d_counts = thrust::raw_pointer_cast(counts.data());
  int* d_input  = thrust::raw_pointer_cast(input.data()) + offset;

  device_for_each(d_input, d_input + num_items, incrementer_t{d_counts});

  const int num_of_once_marked_items =
    static_cast<int>(thrust::count(c2h::device_policy, counts.begin(), counts.end(), 1));

  REQUIRE(num_of_once_marked_items == num_items);
}

using offset_type = c2h::type_list<std::int32_t, std::uint32_t, std::uint64_t, std::int64_t>;

C2H_TEST("Device for each n works", "[for][device]", offset_type)
{
  using offset_t = c2h::get<0, TestType>;

  constexpr int max_items = 5000000;
  constexpr int min_items = 1;

  const auto num_items = static_cast<offset_t>(GENERATE_COPY(
    take(3, random(min_items, max_items)),
    values({
      min_items,
      max_items,
    })));

  c2h::device_vector<offset_proxy_t<offset_t>> input(num_items, offset_t{});
  thrust::sequence(c2h::device_policy, input.begin(), input.end(), offset_t{});

  c2h::device_vector<int> counts(num_items);
  int* d_counts = thrust::raw_pointer_cast(counts.data());

  device_for_each_n(input.begin(), num_items, incrementer_t{d_counts});

  const auto num_of_once_marked_items =
    static_cast<offset_t>(thrust::count(c2h::device_policy, counts.begin(), counts.end(), 1));

  REQUIRE(num_of_once_marked_items == num_items);
}

C2H_TEST("Device for each n works with bad operators", "[for][device]", offset_type)
{
  using offset_t = c2h::get<0, TestType>;

  constexpr int max_items = 5000000;
  constexpr int min_items = 1;

  const auto num_items = static_cast<offset_t>(GENERATE_COPY(
    take(3, random(min_items, max_items)),
    values({
      min_items,
      max_items,
    })));

  const std::size_t magic_value = num_items + 1;
  c2h::device_vector<std::size_t> input(num_items, magic_value);
  const std::size_t* d_input = thrust::raw_pointer_cast(input.data());

  device_for_each_n(input.begin(), num_items, referencing_operator_t{d_input, magic_value});

  REQUIRE(thrust::equal(c2h::device_policy, input.begin(), input.end(), thrust::make_counting_iterator(std::size_t{})));
}

C2H_TEST("Device for each n works with unaligned vectors", "[for][device]", offset_type)
{
  using offset_t = c2h::get<0, TestType>;

  constexpr int max_items = 5000000;
  constexpr int min_items = 1;

  const auto num_items = static_cast<offset_t>(GENERATE_COPY(
    take(3, random(min_items, max_items)),
    values({
      min_items,
      max_items,
    })));

  const int offset = GENERATE(1, 2, 3);

  c2h::device_vector<int> counts(num_items);
  c2h::device_vector<int> input(num_items + offset);
  thrust::sequence(c2h::device_policy, input.begin() + offset, input.end());

  int* d_counts = thrust::raw_pointer_cast(counts.data());
  int* d_input  = thrust::raw_pointer_cast(input.data()) + offset;

  device_for_each_n(d_input, num_items, incrementer_t{d_counts});

  const auto num_of_once_marked_items =
    static_cast<offset_t>(thrust::count(c2h::device_policy, counts.begin(), counts.end(), 1));

  REQUIRE(num_of_once_marked_items == num_items);
}

C2H_TEST("Device for each works with counting iterator", "[for][device]")
{
  using offset_t               = int;
  constexpr offset_t max_items = 5000000;
  constexpr offset_t min_items = 1;
  const offset_t num_items     = GENERATE_COPY(
    take(3, random(min_items, max_items)),
    values({
      min_items,
      max_items,
    }));

  const auto it = thrust::counting_iterator<int>{0};
  c2h::device_vector<int> counts(num_items);
  device_for_each(it, it + num_items, incrementer_t{thrust::raw_pointer_cast(counts.data())});

  const auto num_of_once_marked_items = static_cast<offset_t>(thrust::count(counts.begin(), counts.end(), 1));
  REQUIRE(num_of_once_marked_items == num_items);
}
