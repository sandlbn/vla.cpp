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
 * @file queue_sycl.h
 * @brief Internal access to the process queue owned by @c device_sycl.cpp.
 *
 * Not part of the kernel API - @c device.h deliberately says @c vla_stream and
 * nothing more, so the engine stays free of @c sycl.hpp. The kernel sources are
 * a different matter: they need the live @c sycl::queue to submit against. This
 * is how they ask for it, and it is included by SYCL sources only.
 */

#pragma once

#include "kernels/bitvla/device.h"

#include <sycl/sycl.hpp>

namespace vla {

/**
 * @brief The queue a kernel should submit to.
 * @param stream Null selects the process-wide in-order queue, which is what
 *               every call site in the engine passes.
 * @warning Undefined unless @ref bitvla_sycl_ready is true.
 */
sycl::queue & bitvla_sycl_queue(vla_stream stream);

/** @brief Whether a device context exists yet (i.e. @c vla_dev_set succeeded). */
bool bitvla_sycl_ready();

}  // namespace vla
