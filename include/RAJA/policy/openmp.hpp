/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing RAJA headers for OpenMP execution.
 *
 *          These methods work only on platforms that support OpenMP.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//


#ifndef RAJA_openmp_HPP
#define RAJA_openmp_HPP

#include "RAJA/config.hpp"

#if defined(RAJA_ENABLE_OPENMP)

#include <omp.h>
#include <iostream>
#include <thread>

#include "RAJA/policy/openmp/atomic.hpp"
#include "RAJA/policy/openmp/forall.hpp"
#include "RAJA/policy/openmp/kernel.hpp"
#include "RAJA/policy/openmp/policy.hpp"
#include "RAJA/policy/openmp/reduce.hpp"
#include "RAJA/policy/openmp/region.hpp"
#include "RAJA/policy/openmp/scan.hpp"
#include "RAJA/policy/openmp/synchronize.hpp"

#endif  // closing endif for if defined(RAJA_ENABLE_OPENMP)

#endif  // closing endif for header file include guard
