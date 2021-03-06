/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file containing constructs used to run kernel
 *          traversals on GPU with CUDA.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_policy_cuda_kernel_Lambda_HPP
#define RAJA_policy_cuda_kernel_Lambda_HPP

#include "RAJA/config.hpp"

#if defined(RAJA_ENABLE_CUDA)

#include <cassert>
#include <climits>

#include "camp/camp.hpp"

#include "RAJA/util/macros.hpp"
#include "RAJA/util/types.hpp"

#include "RAJA/pattern/kernel.hpp"
#include "RAJA/pattern/kernel/Lambda.hpp"


namespace RAJA
{
namespace internal
{

template <typename Data, camp::idx_t LambdaIndex>
struct CudaStatementExecutor<Data, statement::Lambda<LambdaIndex>> {

  static
  inline RAJA_DEVICE void exec(Data &data, bool thread_active)
  {
    // Only execute the lambda if it hasn't been masked off
    if(thread_active){
      invoke_lambda<LambdaIndex>(data);
    }
  }


  static
  inline
  LaunchDims calculateDimensions(Data const & RAJA_UNUSED_ARG(data))
  {
    return LaunchDims();
  }
};

//

template <typename Data, camp::idx_t LambdaIndex, typename... Args>
struct CudaStatementExecutor<Data, statement::Lambda<LambdaIndex, Args...>> {

  static
  inline RAJA_DEVICE void exec(Data &data, bool thread_active)
  {

    //Convert SegList, ParamList into Seg, Param types, and store in a list
    using targList = typename camp::flatten<camp::list<Args...>>::type;

    // Only execute the lambda if it hasn't been masked off
    if(thread_active){
      invoke_lambda_with_args<LambdaIndex, targList>(data);
    }

  }


  static
  inline
  LaunchDims calculateDimensions(Data const & RAJA_UNUSED_ARG(data))
  {
    return LaunchDims();
  }
};




}  // namespace internal
}  // namespace RAJA

#endif  // closing endif for RAJA_ENABLE_CUDA guard

#endif  // closing endif for header file include guard
