/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing RAJA index set and segment iteration
 *          template methods for SYCL.
 *
 *          These methods should work on any platform that supports SYCL.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_forall_sycl_HPP
#define RAJA_forall_sycl_HPP

#include "RAJA/config.hpp"

#if defined(RAJA_ENABLE_SYCL)

#include "CL/sycl.hpp"
namespace sycl = cl::sycl;

#include "RAJA/util/types.hpp"

#include "RAJA/policy/sycl/policy.hpp"

#include "RAJA/index/IndexSet.hpp"
#include "RAJA/index/ListSegment.hpp"
#include "RAJA/index/RangeSegment.hpp"

#include "RAJA/internal/fault_tolerance.hpp"

#include "RAJA/pattern/forall.hpp"


namespace RAJA
{

namespace policy
{
namespace sycl
{


/**
 * @brief SYCL dynamic for implementation
 *
 * @param p sycl tag
 * @param iter any iterable
 * @param loop_body loop body
 *
 * @return None
 *
 *
 * This forall implements a SYCL parallel_for loop over the specified iterable
 * using the dynamic loop scheduler and the grain size specified in the policy
 * argument.  This should be used for composable parallelism and increased work
 * stealing at the cost of initial start-up overhead for a top-level loop.
 */
template <typename Iterable, typename Func>
RAJA_INLINE void forall_impl(const sycl_for_dynamic&,
                             Iterable&& iter,
                             Func&& loop_body)
{
  using std::begin;
  using std::end;

#if 0
  using brange = ::sycl::blocked_range<decltype(iter.begin())>;
  ::sycl::parallel_for(brange(begin(iter), end(iter), p.grain_size),
                      [=](const brange& r) {
                        using RAJA::internal::thread_privatize;
                        auto privatizer = thread_privatize(loop_body);
                        auto body = privatizer.get_priv();
                        for (const auto& i : r)
                          body(i);
                      });
#endif

  auto b = begin(iter);
  auto e = end(iter);

  ::sycl::queue q(::sycl::default_selector{});

  using RAJA::internal::thread_privatize;
  auto privatizer = thread_privatize(loop_body);
  auto& body = privatizer.get_priv();

  q.submit([&](::sycl::handler& h) {
    h.parallel_for( ::sycl::range<1>{e},
                    ::sycl::id<1>{b},
                    [=] (::sycl::id<1> it) {
      const size_t i = it[0];
      body(i);
    });
  });
  q.wait();
}

///
/// SYCL parallel for static policy implementation
///

/**
 * @brief SYCL static for implementation
 *
 * @param sycl_for_static sycl tag
 * @param iter any iterable
 * @param loop_body loop body
 *
 * @return None
 *
 * This forall implements a SYCL parallel_for loop over the specified iterable
 * using the static loop scheduler and the grain size specified as a
 * compile-time constant in the policy argument.  This should be used for
 * OpenMP-like fast-launch well-balanced loops, or loops where the split between
 * threads must be maintained across multiple loops for correctness. NOTE: if
 * correctnes requires the per-thread mapping, you *must* use SYCL 2017 or newer
 */
template <typename Iterable, typename Func, size_t ChunkSize>
RAJA_INLINE void forall_impl(const sycl_for_static<ChunkSize>&,
                             Iterable&& iter,
                             Func&& loop_body)
{
  using std::begin;
  using std::end;

#if 0
  using brange = ::sycl::blocked_range<decltype(iter.begin())>;
  ::sycl::parallel_for(brange(begin(iter), end(iter), ChunkSize),
                      [=](const brange& r) {
                        using RAJA::internal::thread_privatize;
                        auto privatizer = thread_privatize(loop_body);
                        auto body = privatizer.get_priv();
                        for (const auto& i : r)
                          body(i);
                      },
                      sycl_static_partitioner{});
#endif

  auto b = begin(iter);
  auto e = end(iter);

  ::sycl::queue q(::sycl::default_selector{});

  using RAJA::internal::thread_privatize;
  auto privatizer = thread_privatize(loop_body);
  auto& body = privatizer.get_priv();

  q.submit([&](::sycl::handler& h) {
    h.parallel_for( ::sycl::range<1>{e},
                    ::sycl::id<1>{b},
                    [=] (::sycl::id<1> it) {
      const size_t i = it[0];
      body(i);
    });
  });
  q.wait();
}

}  // namespace sycl
}  // namespace policy

}  // namespace RAJA

#endif  // closing endif for if defined(RAJA_ENABLE_SYCL)

#endif  // closing endif for header file include guard
