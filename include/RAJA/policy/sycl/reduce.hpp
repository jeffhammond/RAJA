/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing RAJA reduction templates for
 *          SYCL execution.
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

#ifndef RAJA_sycl_reduce_HPP
#define RAJA_sycl_reduce_HPP

#include "RAJA/config.hpp"

#if defined(RAJA_ENABLE_SYCL)

#include <memory>
#include <tuple>

#include <sycl/sycl.h>

#include "RAJA/internal/MemUtils_CPU.hpp"

#include "RAJA/pattern/detail/reduce.hpp"
#include "RAJA/pattern/reduce.hpp"

#include "RAJA/policy/sycl/policy.hpp"

#include "RAJA/util/types.hpp"

namespace RAJA
{

namespace detail
{
template <typename T, typename Reduce>
class ReduceSYCL
{
  //! SYCL native per-thread container
  std::shared_ptr<sycl::combinable<T>> data;

public:
  //! default constructor calls the reset method
  ReduceSYCL() { reset(T(), T()); }

  //! constructor requires a default value for the reducer
  explicit ReduceSYCL(T init_val, T initializer)
  {
    reset(init_val, initializer);
  }

  void reset(T init_val, T initializer)
  {
    data = std::shared_ptr<sycl::combinable<T>>(
        std::make_shared<sycl::combinable<T>>([=]() { return initializer; }));
    data->local() = init_val;
  }

  /*!
   *  \return the calculated reduced value
   */
  T get() const { return data->combine(typename Reduce::operator_type{}); }

  /*!
   *  \return update the local value
   */
  void combine(const T& other) { Reduce{}(this->local(), other); }

  /*!
   *  \return reference to the local value
   */
  T& local() { return data->local(); }
};
}  // namespace detail

RAJA_DECLARE_ALL_REDUCERS(sycl_reduce, detail::ReduceSYCL)

}  // namespace RAJA

#endif  // closing endif for RAJA_ENABLE_SYCL guard

#endif  // closing endif for header file include guard
