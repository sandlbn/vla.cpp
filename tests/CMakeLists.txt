# vla.cpp tests, enabled with -DVLA_BUILD_TESTS=ON. predict_check needs a real
# GGUF so it is built but not run by ctest; the unit tests below are pure.

add_executable(vla_predict_check predict_check.cpp)
target_link_libraries(vla_predict_check PRIVATE vla_core)
target_compile_options(vla_predict_check PRIVATE -Wall -Wextra)

add_executable(test_vision_common test_vision_common.cpp)
target_include_directories(test_vision_common PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_compile_options(test_vision_common PRIVATE -Wall -Wextra)
add_test(NAME vision_common COMMAND test_vision_common)

add_executable(test_rope_conventions test_rope_conventions.cpp)
target_include_directories(test_rope_conventions PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_compile_options(test_rope_conventions PRIVATE -Wall -Wextra)
add_test(NAME rope_conventions COMMAND test_rope_conventions)

# Built as C so the public header stays C-clean.
add_executable(test_c_api test_c_api.c)
target_link_libraries(test_c_api PRIVATE vla)
target_compile_options(test_c_api PRIVATE -Wall -Wextra)
add_test(NAME c_api COMMAND test_c_api)

add_executable(test_dit_common test_dit_common.cpp)
target_include_directories(test_dit_common PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(test_dit_common PRIVATE ggml)
target_compile_options(test_dit_common PRIVATE -Wall -Wextra)
add_test(NAME dit_common COMMAND test_dit_common)

# Defines GGML_USE_OPENVINO itself so the pass is exercised on any build.
add_executable(test_graph_names test_graph_names.cpp)
target_include_directories(test_graph_names PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(test_graph_names PRIVATE ggml)
target_compile_options(test_graph_names PRIVATE -Wall -Wextra)
add_test(NAME graph_names COMMAND test_graph_names)

add_executable(test_qwen3vl_vit test_qwen3vl_vit.cpp)
target_include_directories(test_qwen3vl_vit PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(test_qwen3vl_vit PRIVATE ggml)
target_compile_options(test_qwen3vl_vit PRIVATE -Wall -Wextra)
add_test(NAME qwen3vl_vit COMMAND test_qwen3vl_vit)

# Links vla_core: it calls the real config_is_sane rather than a copy.
add_executable(test_config_guard test_config_guard.cpp)
target_include_directories(test_config_guard PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(test_config_guard PRIVATE vla_core)
target_compile_options(test_config_guard PRIVATE -Wall -Wextra)
add_test(NAME config_guard COMMAND test_config_guard)

# A/B harness for the two BitVLA ternary-GEMM tilings. Built so it cannot rot,
# not registered with ctest: it needs a GPU and is read by hand.
# VLA_BITVLA_NARROW_GEMM=1 selects the old one-tile-per-CTA kernel at runtime.
if(GGML_CUDA)
    add_executable(bitvla_gemm_check bitvla_gemm_check.cu)
    target_include_directories(bitvla_gemm_check PRIVATE ${CMAKE_SOURCE_DIR}/src)
    target_link_libraries(bitvla_gemm_check PRIVATE bitvla_cuda_kernels)
endif()

# The ladder int2 layout and its inverse. Pure CPU: the oneDNN GEMM path reads
# published weights through the unpack, so it is worth pinning without a GPU.
add_executable(test_ladder_pack test_ladder_pack.cpp)
target_include_directories(test_ladder_pack PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_compile_options(test_ladder_pack PRIVATE -Wall -Wextra)
add_test(NAME ladder_pack COMMAND test_ladder_pack)

# The Q8_0/Q4_0 ternary encoders in scripts/bitvla_quant.py, dequantised by
# ggml's own C code. ci/test_bitvla_quant.py writes the fixture and checks it
# against the encoders on every run, so the pair cannot go stale in either
# direction. Pure CPU: it reads a checked-in file, no checkpoint and no GPU.
add_executable(test_bitvla_quant test_bitvla_quant.cpp)
target_include_directories(test_bitvla_quant PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(test_bitvla_quant PRIVATE ggml)
target_compile_definitions(test_bitvla_quant PRIVATE
    VLA_BITVLA_QUANT_FIXTURE="${CMAKE_CURRENT_SOURCE_DIR}/data/bitvla_q_blocks.bin")
target_compile_options(test_bitvla_quant PRIVATE -Wall -Wextra)
add_test(NAME bitvla_quant COMMAND test_bitvla_quant)

# BitVLA's activation quantiser through the OpenVINO translator added by hunk 14
# of scripts/patch_ggml_openvino.py. Only ggml is linked: the test builds its own
# one-node graph and compares the backend against ggml's CPU execution of it, so
# it needs no checkpoint. Skips itself (exit 0) when the runtime finds no device.
if(GGML_OPENVINO)
    add_executable(test_bitvla_act_quant_ov test_bitvla_act_quant_ov.cpp)
    target_include_directories(test_bitvla_act_quant_ov PRIVATE ${CMAKE_SOURCE_DIR}/src)
    target_link_libraries(test_bitvla_act_quant_ov PRIVATE ggml)
    target_compile_options(test_bitvla_act_quant_ov PRIVATE -Wall -Wextra)
    add_test(NAME bitvla_act_quant_ov COMMAND test_bitvla_act_quant_ov)
endif()

# CPU replay of the BitVLA bf16 ops. One source, whichever backend is linked:
# the ops are declared in vendor-neutral terms, so the SYCL port and the CUDA
# original are held to the same thresholds by the same code. Skips itself when
# no GPU is present, so ctest can run it anywhere.
if(GGML_SYCL AND NOT GGML_CUDA)
    add_executable(test_bitvla_ops_gpu test_bitvla_ops_gpu.cpp)
    target_include_directories(test_bitvla_ops_gpu PRIVATE ${CMAKE_SOURCE_DIR}/src)
    target_link_libraries(test_bitvla_ops_gpu PRIVATE bitvla_sycl_kernels)
    # The kernels are a -fsycl archive; the link needs the flag to pull in the
    # device images, and the test source itself has no SYCL in it.
    target_link_options(test_bitvla_ops_gpu PRIVATE -fsycl)
    target_compile_options(test_bitvla_ops_gpu PRIVATE -Wall -Wextra)
    add_test(NAME bitvla_ops_gpu COMMAND test_bitvla_ops_gpu)
elseif(GGML_CUDA)
    add_executable(test_bitvla_ops_gpu test_bitvla_ops_gpu.cpp)
    target_include_directories(test_bitvla_ops_gpu PRIVATE ${CMAKE_SOURCE_DIR}/src)
    target_link_libraries(test_bitvla_ops_gpu PRIVATE bitvla_cuda_kernels)
    target_compile_options(test_bitvla_ops_gpu PRIVATE -Wall -Wextra)
    add_test(NAME bitvla_ops_gpu COMMAND test_bitvla_ops_gpu)
endif()

# The ternary GEMM against host arithmetic. Same source for both backends, and
# the same near-exact gate: int32 accumulation is order-independent, so a GPU
# that tiles K differently still owes the identical answer. This is what makes
# the oneDNN path trustworthy as an oracle for the hand-written DPAS kernel.
if(GGML_SYCL AND NOT GGML_CUDA)
    add_executable(test_bitvla_gemm_gpu test_bitvla_gemm_gpu.cpp)
    target_include_directories(test_bitvla_gemm_gpu PRIVATE ${CMAKE_SOURCE_DIR}/src)
    target_link_libraries(test_bitvla_gemm_gpu PRIVATE bitvla_sycl_kernels)
    target_link_options(test_bitvla_gemm_gpu PRIVATE -fsycl)
    target_compile_options(test_bitvla_gemm_gpu PRIVATE -Wall -Wextra)
    add_test(NAME bitvla_gemm_gpu COMMAND test_bitvla_gemm_gpu)
elseif(GGML_CUDA)
    add_executable(test_bitvla_gemm_gpu test_bitvla_gemm_gpu.cpp)
    target_include_directories(test_bitvla_gemm_gpu PRIVATE ${CMAKE_SOURCE_DIR}/src)
    target_link_libraries(test_bitvla_gemm_gpu PRIVATE bitvla_cuda_kernels)
    # --use_fast_math on the kernels makes float division approximate; see the
    # file comment for why that buys exactly one bf16 step of slack.
    target_compile_definitions(test_bitvla_gemm_gpu PRIVATE VLA_GEMM_CHECK_FAST_MATH=1)
    target_compile_options(test_bitvla_gemm_gpu PRIVATE -Wall -Wextra)
    add_test(NAME bitvla_gemm_gpu COMMAND test_bitvla_gemm_gpu)
endif()

# Roofline study for Battlemage: the ternary GEMM against ceilings measured on
# the same device in the same process. Not registered with ctest - it is a
# measurement, not a gate, and a loaded node would make it a flaky one.
if(GGML_SYCL AND NOT GGML_CUDA)
    add_executable(bench_bitvla_bmg bench_bitvla_bmg.cpp)
    target_include_directories(bench_bitvla_bmg PRIVATE ${CMAKE_SOURCE_DIR}/src)
    target_link_libraries(bench_bitvla_bmg PRIVATE bitvla_sycl_kernels)
    # joint_matrix is an experimental extension and warns about it on every use;
    # the preflight already confirmed the combinations this asks for exist.
    target_compile_options(bench_bitvla_bmg PRIVATE
        -fsycl -O3 -Wall -Wextra -Wno-unused-variable
        -Wno-sycl-target -DSYCL_EXT_ONEAPI_MATRIX_VERSION=4)
    target_link_options(bench_bitvla_bmg PRIVATE -fsycl)
endif()

# Regression test for the in-tree BF16 CUDA kernels and the ggml hook they ride
# on. Skips itself (exit 0) when no CUDA device is present.
if(GGML_CUDA)
    add_executable(test_bf16_cuda_ops test_bf16_cuda_ops.cpp)
    target_include_directories(test_bf16_cuda_ops PRIVATE ${CMAKE_SOURCE_DIR}/src)
    target_link_libraries(test_bf16_cuda_ops PRIVATE vla_cuda_ops ggml)
    target_compile_options(test_bf16_cuda_ops PRIVATE -Wall -Wextra)
    add_test(NAME bf16_cuda_ops COMMAND test_bf16_cuda_ops)
endif()

# The same regression test for the SYCL side. Skips itself (exit 0) when no SYCL
# device is present, so a build on a CPU-only node still passes ctest.
if(GGML_SYCL AND NOT GGML_CUDA)
    add_executable(test_bf16_sycl_ops test_bf16_sycl_ops.cpp)
    target_include_directories(test_bf16_sycl_ops PRIVATE ${CMAKE_SOURCE_DIR}/src)
    target_link_libraries(test_bf16_sycl_ops PRIVATE vla_sycl_ops ggml)
    # -fsycl on the link line gathers the device images out of vla_sycl_ops;
    # without it the first kernel launch segfaults inside the program manager
    # rather than failing to link. Same trap bitvla_sycl_kernels documents.
    target_compile_options(test_bf16_sycl_ops PRIVATE -fsycl -Wall -Wextra)
    target_link_options(test_bf16_sycl_ops PRIVATE -fsycl)
    add_test(NAME bf16_sycl_ops COMMAND test_bf16_sycl_ops)
endif()
