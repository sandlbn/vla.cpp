// Copyright 2026 VinRobotics
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/**
 * @file dnnl_sycl.h
 * @brief One oneDNN engine and stream, bound to the queue the kernels use.
 *
 * Both oneDNN call sites - the ternary reference GEMM and the dense attention
 * GEMMs - need an engine, and they must be the *same* engine: oneDNN caches
 * generated kernels per engine, and a second one would mean a second JIT of
 * every primitive. More importantly the stream has to wrap the queue from
 * @c device_sycl.cpp rather than one of oneDNN's own, or the GEMMs would not be
 * ordered against the elementwise kernels around them and the weights - USM
 * allocations against that queue's context - would not be readable.
 *
 * Included by SYCL sources only.
 */

#pragma once

#include "kernels/bitvla/device.h"

#include "oneapi/dnnl/dnnl.hpp"

namespace vla {

/** @brief The process-wide oneDNN engine for @p stream's device. */
dnnl::engine & bitvla_dnnl_engine(vla_stream stream);

/** @brief A oneDNN stream wrapping the same @c sycl::queue the kernels submit to. */
dnnl::stream & bitvla_dnnl_stream(vla_stream stream);

}  // namespace vla
