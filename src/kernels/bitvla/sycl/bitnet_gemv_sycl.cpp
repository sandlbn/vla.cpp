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
 * @file bitnet_gemv_sycl.cpp
 * @brief M=1 ternary GEMV over the 2-bit pack, for LM decode.
 *
 * The oneDNN path in @c bitnet_sycl.cpp unpacks the weights to int8 first, which
 * costs 4x the footprint. On a memory-bound GEMV that is the whole cost, so this
 * kernel reads the pack in place. Two measurements from @c bench_bitvla_bmg
 * shaped it, and neither is the obvious one:
 *
 *  - Xe compresses memory surfaces losslessly, and unpacked ternary weights hold
 *    only four distinct byte values, so the unpacked stream already reads at
 *    ~1180 GB/s against ~595 GB/s for incompressible bytes. The 4x footprint
 *    penalty is therefore only a ~2x bandwidth penalty, and 2x is what this
 *    kernel is playing for - not 4x.
 *  - At M=1 the achieved bandwidth of the oneDNN path tracks N almost exactly:
 *    586 GB/s at N=2560, 840 at N=3840, 1000 at N=13824. N is the only dimension
 *    it parallelises over, and at N=2560 there is not enough of it. So this
 *    kernel must split K as well, or it inherits that ceiling and wins nothing.
 *
 * @section layout Reading the ladder pack without an index computation
 *
 * @c ladder_slot_addr is a chain of divs and mods, but inverting it collapses.
 * With @c slots_per_block = @c LADDER_N_BLOCK*K/16 = K, the in-block slot @c i
 * decomposes into pure bit selects:
 *
 *     n_global = n_block*16 + ((i>>4)&1)*8 + (i&7)
 *     k_base   = (i & ~127) + ((i>>5)&3)*32 + ((i>>3)&1)*16
 *
 * A sub-group of 32 therefore covers one @c LADDER_K_PER_ITER block of 128 slots
 * if each lane takes four of them: @c i = @c 128*jb + @c 4*l + @c m. Because
 * @c m only ever touches bits 0-1, it moves the output row and leaves @c k_base
 * alone, so the four slots a lane loads as one @c uint4 are four different
 * output rows against the *same* 16 activations. One 16-byte weight load, one
 * 16-byte A load, four independent accumulators, and a single shared bias term.
 *
 * The first version of this kernel loaded one @c uint32 per lane and reached
 * only 250-310 GB/s of 594: a lane issued one 4-byte load and then stalled on it
 * through eight @c dp4a. The access was already perfectly coalesced - the
 * shortage was loads in flight, not locality, and that is what the @c uint4
 * buys.
 *
 * @section bias The +2 bias, without __vsubss4
 *
 * Within a slot the four bytes of the word hold k, k+1, k+2, k+3 for group
 * @c g = @c t/4, so @c (w >> 2g) & 0x03030303 is a packed int8x4 of codes lined
 * up against four consecutive activations - one shift and mask, no per-byte
 * extraction. But the codes are biased by +2 and subtracting @c 0x02020202 from
 * a @c uint32 borrows across byte boundaries wherever a code is 0. CUDA sidesteps
 * this with @c __vsubss4 (@c ladder_pack.h:121); SYCL has no such builtin.
 *
 * Rather than reach for one, drop the subtraction:
 *
 *     sum_k A[k]*(code[k] - 2)  ==  sum_k A[k]*code[k]  -  2 * sum_k A[k]
 *
 * Two int32 accumulators, one extra @c dp4a against a constant @c 0x01010101,
 * and no byte-wise unpacking anywhere. It is not an approximation: both sides
 * are integers and both stay far inside int32 (|sum A*code| <= 127*3*K, i.e.
 * 2.6M at the largest K here), so the identity holds exactly and the result is
 * bit-identical to the oneDNN oracle rather than merely close to it.
 */

#include "kernels/bitvla/sycl/bitnet_gemv_sycl.h"

#include "kernels/bitvla/ladder_pack.h"
#include "kernels/bitvla/sycl/queue_sycl.h"
#include "kernels/bitvla/sycl/sycl_compat.h"

#include <sycl/sycl.hpp>

namespace vla {
namespace bitvla {

namespace {

struct k_gemv_packed {};

/// Lanes per sub-group. Fixed by the layout, not tunable - see @ref layout.
constexpr int SG = 32;
/// Slots each lane takes per step, as one uint4.
constexpr int SLOTS_PER_LANE = 4;
/// k-values one sub-group step covers: SG*SLOTS_PER_LANE slots is exactly the
/// pack's own 128-slot iteration block, hence LADDER_K_PER_ITER.
constexpr int K_PER_STEP = LADDER_K_PER_ITER;

static_assert(SG * SLOTS_PER_LANE == LADDER_K_PER_ITER,
              "a sub-group step must be exactly one ladder iteration block");

/**
 * @brief Four int8 products accumulated into int32 - CUDA's @c __dp4a.
 *
 * Both operands arrive packed in a @c uint32 because that is how they come out
 * of memory: @p a is four consecutive activations, @p b four ternary codes.
 */
inline int32_t dp4a(uint32_t a, uint32_t b, int32_t acc) {
#ifdef SYCL_EXT_ONEAPI_DOT_ACCUMULATE
    return sycl::ext::oneapi::experimental::dot_acc(sycl::bit_cast<sycl::vec<int8_t, 4>>(a),
                                                   sycl::bit_cast<sycl::vec<int8_t, 4>>(b), acc);
#else
    // Xe has the DP4A instruction regardless; without the extension macro the
    // compiler has to recognise this pattern, which it does. Kept exact either
    // way - the widening is to int32 before the multiply.
    for (int i = 0; i < 4; ++i) {
        acc += (int32_t) (int8_t) (uint8_t) (a >> (8 * i)) *
               (int32_t) (int8_t) (uint8_t) (b >> (8 * i));
    }
    return acc;
#endif
}

/**
 * @brief How many sub-groups share one 16-row output block, i.e. the K split.
 *
 * The work-group owns 16 outputs however wide it is, so this trades occupancy
 * against a longer SLM reduction. B70 has 256 XVE with 8 threads each, so it
 * wants a couple of thousand sub-groups before it is full; below that, N alone
 * does not fill the machine, which is the failure the oneDNN path shows at
 * N=2560. Only divisors of @p j_total are considered - with so few steps per
 * sub-group, a remainder means some do twice the work of others and the whole
 * group waits on them.
 */
int k_split(int N, int K, int max_sub_groups) {
    const int n_blocks = N / LADDER_N_BLOCK;
    const int j_total  = K / K_PER_STEP;

    int best = 1;
    for (int s = 1; s <= j_total && s <= max_sub_groups; ++s) {
        if (j_total % s != 0) continue;
        best = s;
        if ((long long) n_blocks * s >= 2048) break;
    }
    return best;
}

}  // namespace

bool ternary_gemv_packed(const int8_t * A, const int8_t * packed, vla_bf16 * out, const float * s,
                         const float * ws, int N, int K, int ws_num, vla_stream stream) {
    // The layout reasoning above assumes whole 16-row blocks and whole 32-slot
    // steps. Every BitVLA shape satisfies both; refuse rather than mis-index if
    // a future one does not, and let the caller fall back.
    if (N % LADDER_N_BLOCK != 0 || K % K_PER_STEP != 0 || ws_num <= 0 || N % ws_num != 0) {
        return false;
    }

    sycl::queue & q = vla::bitvla_sycl_queue(stream);

    const size_t max_wg = q.get_device().get_info<sycl::info::device::max_work_group_size>();

    const int    n_blocks  = N / LADDER_N_BLOCK;
    const int    j_total   = K / K_PER_STEP;
    const int    split     = k_split(N, K, (int) (max_wg / SG));
    const int    wg        = split * SG;
    const int    per_group = N / ws_num;

    bf16 * o = as_bf(out);

    q.submit([&](sycl::handler & h) {
        sycl::local_accessor<int32_t, 1> part(sycl::range<1>((size_t) split * LADDER_N_BLOCK), h);

        h.parallel_for<k_gemv_packed>(
            sycl::nd_range<1>(sycl::range<1>((size_t) n_blocks * wg), sycl::range<1>((size_t) wg)),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
                const int lid  = (int) it.get_local_id(0);
                const int lane = lid % SG;
                const int sg   = lid / SG;
                const int nb   = (int) it.get_group(0);

                // Bits 0 and 2 of the lane pick the output rows, bits 1, 3 and 4
                // pick which 16 of the block's 128 k-values this lane covers.
                const int row_base = ((lane >> 2) & 1) * 8 + (lane & 1) * SLOTS_PER_LANE;
                const int k_off    = ((lane >> 3) & 3) * 32 + ((lane >> 1) & 1) * 16;

                const uint32_t * wp = (const uint32_t *) packed + (size_t) nb * K;

                int32_t acc[SLOTS_PER_LANE] = { 0, 0, 0, 0 };  // sum A[k]*code[k], per row
                int32_t acc_a               = 0;  // sum A[k], shared: same k for all four

                // Strided so that at any step the whole work-group is reading one
                // contiguous run of split*512 bytes.
                for (int jb = sg; jb < j_total; jb += split) {
                    const sycl::uint4 w =
                        *(const sycl::uint4 *) (wp + (size_t) jb * K_PER_STEP + SLOTS_PER_LANE * lane);
                    const sycl::uint4 a = *(const sycl::uint4 *) (A + jb * K_PER_STEP + k_off);

#pragma unroll
                    for (int g = 0; g < 4; ++g) {
                        acc_a = dp4a(a[g], 0x01010101u, acc_a);
#pragma unroll
                        for (int m = 0; m < SLOTS_PER_LANE; ++m) {
                            acc[m] = dp4a(a[g], (w[m] >> (2 * g)) & 0x03030303u, acc[m]);
                        }
                    }
                }

                // Undo the +2 bias per lane, before any cross-lane traffic, so
                // what gets reduced is already this lane's true partial.
#pragma unroll
                for (int m = 0; m < SLOTS_PER_LANE; ++m) acc[m] -= 2 * acc_a;

                // Eight lanes share a row group - those differing in bits 1, 3
                // and 4 - so three butterfly rounds land the total in the lane
                // with all three clear. Collective, so every lane participates;
                // the ones whose partner is stale simply do not store.
                auto sgo = it.get_sub_group();
#pragma unroll
                for (int r = 0; r < 3; ++r) {
                    // 2, 8, 16 - each round clears one of the three free bits,
                    // and none of them disturbs a bit an earlier round settled.
                    const int delta = (r == 0) ? 2 : (r == 1 ? 8 : 16);
#pragma unroll
                    for (int m = 0; m < SLOTS_PER_LANE; ++m) {
                        acc[m] += sycl::shift_group_left(sgo, acc[m], delta);
                    }
                }

                if ((lane & 0x1a) == 0) {
#pragma unroll
                    for (int m = 0; m < SLOTS_PER_LANE; ++m) {
                        part[sg * LADDER_N_BLOCK + row_base + m] = acc[m];
                    }
                }

                it.barrier(sycl::access::fence_space::local_space);

                if (lid < LADDER_N_BLOCK) {
                    // Integer addition is exact and order-independent, so the
                    // sum here is the same int32 the oracle accumulates however
                    // K was split. Only then the single divide, multiply and RNE
                    // round, in the order bitnet_kernels.cu writes them.
                    int32_t sum = 0;
                    for (int t = 0; t < split; ++t) sum += part[t * LADDER_N_BLOCK + lid];

                    const int n = nb * LADDER_N_BLOCK + lid;
                    o[n]        = to_bf16(exact_div((float) sum, s[0]) * ws[n / per_group]);
                }
            });
    });

    return true;
}

}  // namespace bitvla
}  // namespace vla
