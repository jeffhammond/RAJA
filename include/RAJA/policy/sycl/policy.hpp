/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing RAJA sequential policy definitions.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef policy_sycl_HPP
#define policy_sycl_HPP

#include "RAJA/policy/PolicyBase.hpp"

#include <cstddef>

namespace RAJA
{
namespace policy
{
namespace sycl
{

//
//////////////////////////////////////////////////////////////////////
//
// Execution policies
//
//////////////////////////////////////////////////////////////////////
//

///
/// Segment execution policies
///

struct sycl_for_dynamic
    : make_policy_pattern_launch_platform_t<Policy::sycl,
                                            Pattern::forall,
                                            Launch::undefined,
                                            Platform::host> {
  std::size_t grain_size;
  sycl_for_dynamic(std::size_t grain_size_ = 1) : grain_size(grain_size_) {}
};


template <std::size_t GrainSize = 1>
struct sycl_for_static : make_policy_pattern_launch_platform_t<Policy::sycl,
                                                              Pattern::forall,
                                                              Launch::undefined,
                                                              Platform::host> {
};

using sycl_for_exec = sycl_for_static<>;

///
/// Index set segment iteration policies
///
using sycl_segit = sycl_for_exec;


///
///////////////////////////////////////////////////////////////////////
///
/// Reduction execution policies
///
///////////////////////////////////////////////////////////////////////
///
struct sycl_reduce : make_policy_pattern_launch_platform_t<Policy::sycl,
                                                          Pattern::reduce,
                                                          Launch::undefined,
                                                          Platform::host> {
};

}  // namespace sycl
}  // namespace policy

using policy::sycl::sycl_for_dynamic;
using policy::sycl::sycl_for_exec;
using policy::sycl::sycl_for_static;
using policy::sycl::sycl_reduce;
using policy::sycl::sycl_segit;

}  // namespace RAJA

#endif
