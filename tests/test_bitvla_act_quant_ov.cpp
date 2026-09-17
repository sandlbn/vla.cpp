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

// BitVLA's activation quantiser, translated. It is the one op in that model's
// graph that is not stock ggml, and hunk 14 of scripts/patch_ggml_openvino.py is
// what gives it an OpenVINO translator - the single thing that had kept BitVLA
// pinned to the CPU backend while the other ten architectures ran on this one.
//
// Driven through the real backend rather than against a hand-written copy of the
// OpenVINO subgraph, deliberately: a copy would pass while drifting away from the
// translator it is meant to pin. What is compared is ggml's own CPU execution of
// the custom op against OpenVINO's translation of the same graph.
//
// On the gate, which is two different gates and needs explaining.
//
// This op rounds to an integer, so its output is a step function of its input: a
// value landing on the wrong side of a .5 boundary moves an integer, not a
// low-order bit. A 1-ULP error in the 127/amax scale that feeds it was a shipping
// bug on both the CUDA and the SYCL kernels (see vla_exact_div in
// src/kernels/bitvla/cuda_compat.h), so bit-exactness is what one would want here.
//
// It is not available. OpenVINO's CPU plugin does not have a correctly-rounded f32
// division: Divide(a, b) is emitted as a * (1/b), two roundings where IEEE asks for
// one, and measuring it over 20000 random divisors gives 20000/20000 agreement with
// a * (1/b) and 15007/20000 with the correctly-rounded quotient. That is the emitter
// and not a fusion artefact - disabling snippets does not change it, and neither
// does an explicitly non-constant numerator. Asking for the divide in f64, where the
// extra bits would make the double rounding provably harmless, does not help either:
// the plugin demotes f64 to f32 before it runs. So 127/amax is about one ULP out on
// roughly half of all rows, and there is no formulation of this subgraph that fixes
// it. Divide is still written literally below rather than as the Multiply the plugin
// will turn it into, so that a plugin which one day rounds it correctly makes this
// exact for free.
//
// What that leaves is worth having, so the cases split:
//
//   Gate::Exact - the all-zero row, the sub-floor row and the exact-.5 row. Each has
//     a scale the inexact divide happens to get right (1e-5 and 127 both do), so
//     there is no excuse for any difference at all. These are also the cases that
//     carry the real semantics: the 1e-5 floor, and HALF_TO_EVEN versus
//     HALF_AWAY_FROM_ZERO. A wrong rounding mode moves half of case 3.
//
//   Gate::Step - the random cases, measured in units of the row's own quantisation
//     step rather than in ULPs, because that is the unit the op actually works in.
//     Nothing may be off by more than one step, and the fraction of elements that
//     move by a step at all - an integer that flipped, the only way this op can
//     really be wrong - must stay under 1e-3. A wrong reduction axis, a missing
//     keep_dims or a dropped floor all fail this by orders of magnitude; the
//     plugin's divide does not.
//
// Mutation-tested, because a gate that cannot fail is not a gate. Rounding the
// wrong way moves 14 of case 3's 32 values; reducing over the wrong axis fails
// three cases with deltas up to 127 steps; dropping the 1e-5 floor fails two. What
// it does NOT catch is dequantising by amax/127 instead of by a separately rounded
// 1/s - one rounding where the reference has two, worth about a ULP. That is the
// same magnitude as the plugin's own divide error, so no gate can separate them
// here, and the end-to-end check against the ggml CPU backend is what covers it.
//
// Skips itself (exit 0) when no OpenVINO device is present, so ctest passes on a
// build without the runtime.

#include "backend.h"

#include <ggml-backend.h>
#include <ggml.h>

#undef NDEBUG  // keep assert() live even in Release builds
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

// A copy of bitvla_act_quant_op (src/models/bitvla.cpp), which lives in an
// anonymous namespace and cannot be linked against. Keep the two in step: this is
// the reference the translator is held to, so a divergence here would quietly
// relax the test rather than fail it.
static void act_quant_ref(ggml_tensor * dst, const ggml_tensor * a, int ith, int nth, void *) {
    const int64_t cols = a->ne[0];
    const int64_t rows = ggml_nrows(a);
    const int64_t per  = (rows + nth - 1) / nth;
    const int64_t r0   = ith * per;
    const int64_t r1   = std::min(rows, r0 + per);
    const float * src = (const float *) a->data;
    float * out = (float *) dst->data;
    for (int64_t r = r0; r < r1; ++r) {
        const float * row_in = src + r * cols;
        float * row_out = out + r * cols;
        float amax = 0.0f;
        for (int64_t c = 0; c < cols; ++c)
            amax = std::max(amax, std::fabs(row_in[c]));
        if (amax < 1e-5f)
            amax = 1e-5f;
        const float s     = 127.0f / amax;
        const float inv_s = 1.0f / s;
        for (int64_t c = 0; c < cols; ++c) {
            float q = std::nearbyintf(row_in[c] * s);
            if (q >  127.0f) q =  127.0f;
            if (q < -128.0f) q = -128.0f;
            row_out[c] = q * inv_s;
        }
    }
}

namespace {

int failures = 0;

// One act_quant node over `in`, computed on `be`. The name is the whole point:
// GGML_OP_MAP_CUSTOM1 identifies its kernel by a host function pointer, which
// means nothing to another backend, so the translator keys on the name instead.
bool compute(ggml_backend_t be, const std::vector<float> & in, int64_t rows, int64_t cols,
             std::vector<float> & out) {
    ggml_init_params p = { ggml_tensor_overhead() * 32 + ggml_graph_overhead(), nullptr, true };
    ggml_context * C = ggml_init(p);
    assert(C);

    ggml_tensor * x = ggml_new_tensor_2d(C, GGML_TYPE_F32, cols, rows);
    ggml_set_name(x, "x");
    ggml_set_input(x);

    ggml_tensor * q = ggml_map_custom1(C, x, act_quant_ref, GGML_N_TASKS_MAX, nullptr);
    ggml_set_name(q, "bitvla.act_quant");
    ggml_set_output(q);

    ggml_cgraph * gf = ggml_new_graph(C);
    ggml_build_forward_expand(gf, q);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(C, be);
    if (!buf) {
        ggml_free(C);
        return false;
    }
    ggml_backend_tensor_set(x, in.data(), 0, in.size() * sizeof(float));

    // What every arch does before handing a graph to this backend; the translator
    // tests the name as a substring precisely because this may decorate it.
    vla::graph_unique_names(gf);

    const bool ok = ggml_backend_graph_compute(be, gf) == GGML_STATUS_SUCCESS;
    if (ok) {
        out.resize(in.size());
        ggml_backend_tensor_get(q, out.data(), 0, out.size() * sizeof(float));
    }
    ggml_backend_buffer_free(buf);
    ggml_free(C);
    return ok;
}

enum class Gate { Exact, Step };

void check(ggml_backend_t ov, ggml_backend_t cpu, const char * what, Gate gate,
           const std::vector<float> & in, int64_t rows, int64_t cols) {
    std::vector<float> want, got;
    if (!compute(cpu, in, rows, cols, want)) {
        std::printf("%-30s FAILED: CPU backend did not compute the graph\n", what);
        ++failures;
        return;
    }
    if (!compute(ov, in, rows, cols, got)) {
        std::printf("%-30s FAILED: OpenVINO backend did not compute the graph\n", what);
        ++failures;
        return;
    }

    int64_t n_bitdiff = 0;
    int64_t n_flip    = 0;    // an integer moved: |delta| is a whole quantisation step
    double  max_steps = 0.0;  // the largest delta, in units of that row's step
    for (int64_t r = 0; r < rows; ++r) {
        // The row's own quantisation step, from the reference: a delta means nothing
        // until it is measured against the grid this op rounds onto.
        float amax = 0.0f;
        for (int64_t c = 0; c < cols; ++c)
            amax = std::max(amax, std::fabs(in[r * cols + c]));
        if (amax < 1e-5f)
            amax = 1e-5f;
        const double step = 1.0 / (double) (127.0f / amax);

        for (int64_t c = 0; c < cols; ++c) {
            const size_t i = (size_t) (r * cols + c);
            uint32_t a, b;
            std::memcpy(&a, &got[i],  4);
            std::memcpy(&b, &want[i], 4);
            n_bitdiff += (a != b);
            const double d = std::fabs((double) got[i] - (double) want[i]) / step;
            n_flip    += (d >= 0.5);
            max_steps  = std::max(max_steps, d);
        }
    }

    const double flip_frac = (double) n_flip / (double) in.size();
    const bool   ok = gate == Gate::Exact ? n_bitdiff == 0
                                          : (max_steps <= 1.001 && flip_frac <= 1e-3);
    std::printf("%-30s %-5s rows=%-4lld cols=%-5lld  differing %7lld/%-7zu"
                "  max %.2e step  flipped %lld (%.2e)  %s\n",
                what, gate == Gate::Exact ? "exact" : "step", (long long) rows,
                (long long) cols, (long long) n_bitdiff, in.size(), max_steps,
                (long long) n_flip, flip_frac, ok ? "ok" : "FAILED");
    if (!ok)
        ++failures;
}

}  // namespace

int main() {
    // Directly rather than through vla::backend_init, which falls back to the CPU
    // backend when OpenVINO will not start -- that fallback would turn this into a
    // CPU-against-CPU comparison that passes without testing anything.
    ggml_backend_t ov = ggml_backend_openvino_init(0);
    if (!ov) {
        std::printf("no OpenVINO device, skipping\n");
        return 0;
    }
    ggml_backend_t cpu = ggml_backend_cpu_init();
    assert(cpu);

    // 1. All zeros. amax is 0, the 1e-5 floor is what stops the divide, and every
    //    output must be 0 rather than NaN.
    check(ov, cpu, "all-zero row (1e-5 floor)", Gate::Exact,
          std::vector<float>(4 * 128, 0.0f), 4, 128);

    // 2. Magnitudes below the floor. The row must NOT be rescaled to the full int8
    //    range - getting that wrong amplifies near-dead rows, quietly.
    {
        std::vector<float> v(4 * 128);
        for (size_t i = 0; i < v.size(); ++i)
            v[i] = (float) ((int) (i % 7) - 3) * 1e-7f;
        check(ov, cpu, "sub-1e-5 row (floor active)", Gate::Exact, v, 4, 128);
    }

    // 3. Exact .5 boundaries. amax is 127 here, so the scale is exactly 1.0 and
    //    Round sees the inputs themselves: HALF_TO_EVEN sends 0.5 to 0 and 1.5 to
    //    2, HALF_AWAY_FROM_ZERO sends them to 1 and 2. This is the case that tells
    //    the two rounding modes apart, and ggml's nearbyintf is the first.
    {
        const int64_t cols   = 16;
        const float   half[] = { 0.5f, 1.5f, 2.5f, 3.5f, -0.5f, -1.5f, -2.5f, -3.5f,
                                 4.5f, 5.5f, 6.5f, 7.5f, -4.5f, -5.5f, 127.0f, -127.0f };
        std::vector<float> v(2 * cols);
        for (int64_t r = 0; r < 2; ++r)
            for (int64_t c = 0; c < cols; ++c)
                v[r * cols + c] = half[c] * (r ? -1.0f : 1.0f);
        check(ov, cpu, "exact .5 (HALF_TO_EVEN)", Gate::Exact, v, 2, cols);
    }

    // 4. A different scale on every row, spanning 2^-20 to 2^19, so the reduction
    //    has to be genuinely per-row and the reciprocal has to hold up across the
    //    exponent range rather than near 1.
    {
        const int64_t rows = 32, cols = 896;  // BitVLA's LM hidden size
        std::vector<float> v(rows * cols);
        std::mt19937 rng(20260916);
        for (int64_t r = 0; r < rows; ++r) {
            const float mag = std::ldexp(1.0f, (int) (r % 40) - 20);
            std::uniform_real_distribution<float> d(-mag, mag);
            for (int64_t c = 0; c < cols; ++c)
                v[r * cols + c] = d(rng);
        }
        check(ov, cpu, "per-row scale 2^-20..2^19", Gate::Step, v, rows, cols);
    }

    // 5. A ViT-shaped block of plausible activations.
    {
        const int64_t rows = 257, cols = 1536;
        std::vector<float> v(rows * cols);
        std::mt19937 rng(7);
        std::normal_distribution<float> d(0.0f, 0.3f);
        for (auto & x : v) x = d(rng);
        check(ov, cpu, "random normal, ViT-shaped", Gate::Step, v, rows, cols);
    }

    // 6. MAP_CUSTOM1 is ggml's generic escape hatch, not an op: the next model to
    //    use it for something else must fall back to the CPU, not be mistranslated
    //    into a fake-quant or thrown at. The scheduler decides that by asking
    //    supports_op, so that is what is asked here.
    {
        ggml_init_params p = { ggml_tensor_overhead() * 8 + ggml_graph_overhead(), nullptr, true };
        ggml_context * C = ggml_init(p);
        assert(C);
        ggml_tensor * x = ggml_new_tensor_2d(C, GGML_TYPE_F32, 64, 4);

        ggml_tensor * mine = ggml_map_custom1(C, x, act_quant_ref, GGML_N_TASKS_MAX, nullptr);
        ggml_set_name(mine, "bitvla.act_quant");
        ggml_tensor * theirs = ggml_map_custom1(C, x, act_quant_ref, GGML_N_TASKS_MAX, nullptr);
        ggml_set_name(theirs, "somebody.elses_custom_op");

        const bool claims_mine   = ggml_backend_supports_op(ov, mine);
        const bool claims_theirs = ggml_backend_supports_op(ov, theirs);
        std::printf("%-30s claims bitvla.act_quant=%d  claims foreign custom op=%d  %s\n",
                    "MAP_CUSTOM1 name guard", (int) claims_mine, (int) claims_theirs,
                    (claims_mine && !claims_theirs) ? "ok" : "WRONG");
        if (!claims_mine || claims_theirs)
            ++failures;
        ggml_free(C);
    }

    ggml_backend_free(cpu);
    ggml_backend_free(ov);

    std::printf("\n%s\n", failures ? "FAILED" : "act_quant translates within gate");
    return failures ? 1 : 0;
}
