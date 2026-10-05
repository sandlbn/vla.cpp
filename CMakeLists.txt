cmake_minimum_required(VERSION 3.22)
project(vla_cpp LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

set(_vla_accel "")
foreach(_flag GGML_CUDA GGML_SYCL GGML_METAL GGML_OPENVINO)
    if(${_flag})
        list(APPEND _vla_accel ${_flag})
    endif()
endforeach()
list(LENGTH _vla_accel _vla_accel_n)
if(_vla_accel_n GREATER 1)
    string(REPLACE ";" ", " _vla_accel_str "${_vla_accel}")
    message(FATAL_ERROR
        "Enable one accelerator backend at a time; got ${_vla_accel_str}. "
        "Configure a separate build directory per backend.")
endif()
# backend.h calls ggml_backend_<accel>_init directly, and GGML_BACKEND_DL builds
# the accelerators as loadable modules that are not linked into ggml. Say so here
# rather than at the undefined symbol.
if(_vla_accel_n GREATER 0 AND GGML_BACKEND_DL)
    message(FATAL_ERROR
        "GGML_BACKEND_DL=ON is not supported with ${_vla_accel}: the backend is "
        "built as a module and src/backend.h links its init directly.")
endif()

set(LLAMA_BUILD_COMMON   ON  CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS    ON  CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER   OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(_vla_llama_patch "")
if(GGML_CUDA)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    set(_vla_llama_patch PATCH_COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/scripts/patch_ggml_cuda_ext_hook.py <SOURCE_DIR>)
elseif(GGML_SYCL)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    set(_vla_llama_patch PATCH_COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/scripts/patch_ggml_sycl_ext_hook.py <SOURCE_DIR>)
elseif(GGML_OPENVINO)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    set(_vla_llama_patch PATCH_COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/scripts/patch_ggml_openvino.py <SOURCE_DIR>)

    # Before the fetch: ggml's own ggml-openvino/CMakeLists.txt does
    # find_package(OpenVINO REQUIRED ...) and would get there first, so this
    # message only reaches anyone if it runs earlier. Same components it asks for.
    find_package(OpenVINO QUIET COMPONENTS Runtime Threading)
    if(NOT OpenVINO_FOUND)
        message(FATAL_ERROR
            "GGML_OPENVINO=ON but the OpenVINO runtime was not found. Install it "
            "(scripts/install_ov.sh) and 'source /opt/intel/openvino/setupvars.sh' "
            "in the shell that configures. See docs/backend/ov.md.")
    endif()
endif()
# Overridable so a regression can be bisected against another tag in a separate
# build dir (-DVLA_LLAMA_TAG=b10326) without editing this file. The patch
# anchors in scripts/patch_ggml_cuda_ext_hook.py and scripts/patch_ggml_openvino.py
# are checked against the default.
set(_vla_llama_tag_default "b10729")
set(VLA_LLAMA_TAG "${_vla_llama_tag_default}" CACHE STRING "llama.cpp tag to fetch")
# A cache entry survives an edit to the line above, so an existing build dir keeps
# the tag it was first configured with and quietly builds the wrong llama.cpp.
if(NOT VLA_LLAMA_TAG STREQUAL _vla_llama_tag_default)
    message(WARNING
        "VLA_LLAMA_TAG is ${VLA_LLAMA_TAG}, not the default ${_vla_llama_tag_default}. "
        "If that was not deliberate, this build dir predates the bump: reconfigure with "
        "-DVLA_LLAMA_TAG=${_vla_llama_tag_default} or use a fresh one.")
endif()

# Ahead-of-time device code for the SYCL build. Empty (the default) ships
# SPIR-V only and lets the driver JIT every kernel at first use; a device list
# in ocloc's spelling compiles native binaries for those parts at build time and
# keeps SPIR-V alongside for any other GPU. e.g. Panther Lake: "ptl-u,ptl-h";
# Battlemage: "bmg". Forwarded to ggml-sycl's GGML_SYCL_DEVICE_ARCH unless that
# is set explicitly. See docs/backend/ptl.md.
set(VLA_SYCL_DEVICE_ARCH "" CACHE STRING "ocloc device list for SYCL AOT (empty = JIT)")
if(GGML_SYCL AND VLA_SYCL_DEVICE_ARCH AND NOT GGML_SYCL_DEVICE_ARCH)
    set(GGML_SYCL_DEVICE_ARCH "${VLA_SYCL_DEVICE_ARCH}" CACHE STRING "" FORCE)
endif()

include(FetchContent)
FetchContent_Declare(llama
    GIT_REPOSITORY https://github.com/ggml-org/llama.cpp
    GIT_TAG        ${VLA_LLAMA_TAG}
    GIT_SHALLOW    TRUE
    ${_vla_llama_patch}
)
FetchContent_MakeAvailable(llama)

# Drop the whole fetched tree from the default build: llama.cpp's ~20 tool
# binaries and their impl libraries are dead weight here. What we link (llama,
# ggml, mtmd, llama-common) is still built, pulled in as a dependency. Walking
# the tree rather than naming targets keeps a bump from adding them back. Any of
# them builds on request: cmake --build build --target llama-mtmd-cli
function(vla_exclude_fetched_targets dir)
    get_property(subdirs DIRECTORY ${dir} PROPERTY SUBDIRECTORIES)
    foreach(sub IN LISTS subdirs)
        vla_exclude_fetched_targets(${sub})
    endforeach()
    get_property(targets DIRECTORY ${dir} PROPERTY BUILDSYSTEM_TARGETS)
    foreach(tgt IN LISTS targets)
        set_target_properties(${tgt} PROPERTIES EXCLUDE_FROM_ALL TRUE)
    endforeach()
endfunction()
vla_exclude_fetched_targets(${llama_SOURCE_DIR})

add_library(vla_core
    src/model.cpp
    src/loader.cpp
    src/options.cpp
    src/modules/action_expert.cpp
    src/modules/dit_head.cpp
    src/modules/encoder.cpp
    src/modules/prompt.cpp
    src/modules/qwen3_lm.cpp
    src/modules/siglip_vit.cpp
    src/models/smolvla.cpp
    src/models/pi0.cpp
    src/models/pi05.cpp
    src/models/evo1.cpp
    src/models/gr00tn1d5.cpp
    src/models/gr00tn1d6.cpp
    src/models/gr00tn1d7.cpp
    src/models/bitvla.cpp
    src/models/vla_adapter.cpp
    src/models/openvla_oft.cpp
    src/models/vla_jepa.cpp
)
target_include_directories(vla_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${llama_SOURCE_DIR}/vendor
)
# The VLA archs call no llama_* API; only vlm_core needs llama.
target_link_libraries(vla_core PUBLIC ggml)

# BitVLA's LM / ViT / action-head forward passes. Plain C++ against device.h,
# gemm.h and the op headers, so both the CUDA and the SYCL kernel target compile
# the same three files and the layer order exists in exactly one place.
set(VLA_BITVLA_DRIVER_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/bitvla/bitvla_lm_driver.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/bitvla/bitvla_vit_driver.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/bitvla/bitvla_fp32head_driver.cpp
)

if(GGML_CUDA)
    target_compile_definitions(vla_core PUBLIC GGML_USE_CUDA)

    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        # SASS per supported GPU, PTX only for the newest arch so future cards JIT.
        set(_vla_cuda_archs 80-real 86-real 87-real 89-real 90-real)
        set(_vla_cuda_ptx 90-virtual)
        if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL 12.8)
            list(APPEND _vla_cuda_archs 100-real 120-real)
            set(_vla_cuda_ptx 120-virtual)
        else()
            message(WARNING
                "CUDA ${CUDAToolkit_VERSION} < 12.8: omitting Blackwell sm_100/sm_120. "
                "Pass -DCMAKE_CUDA_ARCHITECTURES=<arch> explicitly to override.")
        endif()
        set(CMAKE_CUDA_ARCHITECTURES ${_vla_cuda_archs} ${_vla_cuda_ptx}
            CACHE STRING "" FORCE)
    endif()
    # The three forward drivers are .cpp and are listed once, below, for both
    # backends: they call device.h/gemm.h/the op headers and nothing else, so
    # there is no reason for nvcc to see them.
    add_library(bitvla_cuda_kernels STATIC
        src/kernels/bitvla/bitnet_kernels.cu
        src/kernels/bitvla/bitvla_ops_cuda.cu
        src/kernels/bitvla/bitvla_fp32_ops_cuda.cu
        src/kernels/bitvla/device_cuda.cu
        src/kernels/bitvla/gemm_cuda.cu
        ${VLA_BITVLA_DRIVER_SOURCES}
    )
    set_target_properties(bitvla_cuda_kernels PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON
    )
    target_include_directories(bitvla_cuda_kernels PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_compile_features(bitvla_cuda_kernels PRIVATE cxx_std_17)
    target_compile_options(bitvla_cuda_kernels PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math -Xptxas=-O3>
    )
    target_link_libraries(bitvla_cuda_kernels PUBLIC CUDA::cublas CUDA::cudart)
    target_link_libraries(vla_core PRIVATE bitvla_cuda_kernels)
    # The engine only asks whether hand-written GPU kernels exist; which backend
    # provides them is the business of src/kernels/bitvla/device.h.
    target_compile_definitions(vla_core PUBLIC VLA_BITVLA_GPU_KERNELS)
    target_include_directories(vla_core PUBLIC ${CUDAToolkit_INCLUDE_DIRS})

    # ggml-cuda link: the hook resolves ggml_cuda_ext_forward out of it.
    add_library(vla_cuda_ops STATIC src/cuda/vla_cuda_bf16.cu)
    target_include_directories(vla_cuda_ops PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${llama_SOURCE_DIR}/ggml/include
    )
    set_target_properties(vla_cuda_ops PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON
    )
    target_compile_features(vla_cuda_ops PRIVATE cxx_std_17)
    target_compile_options(vla_cuda_ops PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-O3 -Xptxas=-O3>
    )
    target_link_libraries(vla_cuda_ops PUBLIC CUDA::cublas CUDA::cudart ggml-cuda)
    target_link_libraries(vla_core PRIVATE vla_cuda_ops)
endif()

if(GGML_SYCL AND NOT GGML_CUDA)
    if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
        message(FATAL_ERROR
            "GGML_SYCL=ON needs the oneAPI DPC++ compiler, but CMAKE_CXX_COMPILER is "
            "'${CMAKE_CXX_COMPILER_ID}'. Source /opt/intel/oneapi/setvars.sh and configure a "
            "fresh build dir with -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx. "
            "See docs/backend/sycl.md.")
    endif()
    target_compile_definitions(vla_core PUBLIC GGML_USE_SYCL)

    # BitVLA's ternary stack on Battlemage. Mirrors bitvla_cuda_kernels above and
    # implements the same extern "C" symbols, so src/models/bitvla.cpp is shared.
    #
    # The ternary GEMM's reference path is dnnl::matmul, so oneDNN is required
    # rather than optional here. ggml-sycl treats it as optional and quietly
    # loses performance without it; BitVLA would lose a correctness oracle, so
    # this one fails the configure instead. It ships in the same oneAPI release
    # as icpx, so if the compiler was found this will be too.
    find_package(DNNL REQUIRED)

    # VLA_SYCL_DEVICE_ARCH -> native code for the listed devices plus SPIR-V.
    # The device link happens wherever the archive is linked, so the targets go
    # on the PUBLIC link options too.
    function(vla_sycl_aot tgt)
        if(VLA_SYCL_DEVICE_ARCH)
            set(_aot -fsycl-targets=spir64_gen,spir64
                     "SHELL:-Xsycl-target-backend=spir64_gen \"-device ${VLA_SYCL_DEVICE_ARCH}\"")
            target_compile_options(${tgt} PRIVATE ${_aot})
            target_link_options(${tgt} PUBLIC ${_aot})
        endif()
    endfunction()

    add_library(bitvla_sycl_kernels STATIC
        src/kernels/bitvla/sycl/device_sycl.cpp
        src/kernels/bitvla/sycl/bitvla_ops_sycl.cpp
        src/kernels/bitvla/sycl/bitvla_fp32_ops_sycl.cpp
        src/kernels/bitvla/sycl/bitnet_sycl.cpp
        src/kernels/bitvla/sycl/bitnet_gemv_sycl.cpp
        src/kernels/bitvla/sycl/gemm_sycl.cpp
        src/kernels/bitvla/sycl/bitvla_fused_sycl.cpp
        src/kernels/bitvla/sycl/attention_sycl.cpp
        ${VLA_BITVLA_DRIVER_SOURCES}
    )
    set_target_properties(bitvla_sycl_kernels PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_include_directories(bitvla_sycl_kernels PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_compile_features(bitvla_sycl_kernels PRIVATE cxx_std_17)
    # The shared drivers compile into this target, so this switches them onto
    # the fused row kernels of bitvla_fused.h (only SYCL implements them).
    target_compile_definitions(bitvla_sycl_kernels PRIVATE VLA_BITVLA_FUSED_OPS)
    # -fp-model=precise because icpx, unlike gcc and clang, defaults to
    # -fp-model=fast. Leaving it off is opting *in* to reassociation, so "we do
    # not pass -ffast-math" is not the same statement here that it is elsewhere.
    #
    # The two -foffload-fp32-prec-* flags are kept as a statement of intent, but
    # be clear about what they did not do. SYCL float division is permitted ~2.5
    # ULP of error by default and on B70 it takes it: act_quant's 127.0f/amax
    # came out exactly 1 ULP high (0x42295556 against 0x42295555) on 58 of 61
    # rows. Neither of these flags moved it, and neither did the runtime's
    # SYCL_PROGRAM_COMPILE_OPTIONS=-cl-fp32-correctly-rounded-divide-sqrt. What
    # fixed it was doing the division in source - see exact_div in
    # sycl/sycl_compat.h - which has the further advantage of not depending on a
    # build flag to stay correct.
    #
    # Not one quantised int8 had changed, so this could have been waved through
    # as a rounding detail. It could not: the ternary GEMM's epilogue divides
    # too, and the contract there is bit-exactness against host arithmetic, so
    # 1 ULP is the whole claim rather than a detail of it.
    target_compile_options(bitvla_sycl_kernels PRIVATE
        -O3 -fsycl -fp-model=precise -foffload-fp32-prec-div -foffload-fp32-prec-sqrt)
    # PUBLIC, not PRIVATE. The device images live in this static archive, and
    # whoever links the archive has to run the SYCL device link step to gather
    # them into the final binary. libvla_core.so was linked without -fsycl, so
    # its device images were silently absent and the first kernel launched
    # through it - bitvla_add_bias_bf16, the first one the ViT forward reaches -
    # segfaulted inside ProgramManager::getDeviceKernelInfo. The unit tests did
    # not catch it because they link this archive into an executable and pass
    # -fsycl themselves.
    target_link_options(bitvla_sycl_kernels PUBLIC -fsycl)
    vla_sycl_aot(bitvla_sycl_kernels)
    target_link_libraries(bitvla_sycl_kernels PUBLIC DNNL::dnnl)
    target_link_libraries(vla_core PRIVATE bitvla_sycl_kernels)
    # Same definition the CUDA branch sets: the engine only asks whether
    # hand-written GPU kernels exist. Set here only now that every symbol
    # src/models/bitvla.cpp reaches for has a SYCL definition - while the set was
    # still incomplete, leaving it off gave a clear "no kernels, running on CPU"
    # message at load time instead of a link error.
    target_compile_definitions(vla_core PUBLIC VLA_BITVLA_GPU_KERNELS)

    # ggml-sycl link: the hook resolves ggml_sycl_ext_forward out of it. Mirrors
    # vla_cuda_ops above, and implements the same idea with a quarter of the
    # code - ggml-sycl already has BF16 add/mul/unary, so only norm, rms_norm,
    # scale and the BF16-dst matmul are left. See src/sycl/vla_sycl_bf16.cpp.
    add_library(vla_sycl_ops STATIC src/sycl/vla_sycl_bf16.cpp src/sycl/vla_sycl_attn.cpp)
    target_include_directories(vla_sycl_ops PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${llama_SOURCE_DIR}/ggml/include
    )
    set_target_properties(vla_sycl_ops PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_features(vla_sycl_ops PRIVATE cxx_std_17)
    # Same -fp-model=precise reasoning as bitvla_sycl_kernels above: icpx defaults
    # to -fp-model=fast, so leaving it off is opting *in* to reassociation.
    target_compile_options(vla_sycl_ops PRIVATE
        -O3 -fsycl -fp-model=precise -foffload-fp32-prec-div -foffload-fp32-prec-sqrt)
    # PUBLIC for the same reason bitvla_sycl_kernels needs it: the device images
    # live in this archive and whoever links it must run the SYCL device link.
    target_link_options(vla_sycl_ops PUBLIC -fsycl)
    vla_sycl_aot(vla_sycl_ops)
    target_link_libraries(vla_sycl_ops PUBLIC DNNL::dnnl ggml ggml-sycl)
    target_link_libraries(vla_core PRIVATE vla_sycl_ops)
endif()

# Precedence matches the ladder in src/backend.h.
if(GGML_METAL AND NOT GGML_CUDA AND NOT GGML_SYCL)
    target_compile_definitions(vla_core PUBLIC GGML_USE_METAL)
endif()

if(GGML_OPENVINO AND NOT GGML_CUDA AND NOT GGML_SYCL AND NOT GGML_METAL)
    # Only tells the archs which branch of the backend.h ladder to compile; the
    # toolkit was located before the fetch above.
    target_compile_definitions(vla_core PUBLIC GGML_USE_OPENVINO)
endif()

add_library(vlm_core
    src/vlm/engine.cpp
)
target_include_directories(vlm_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/src
    PRIVATE
        ${llama_SOURCE_DIR}/common
        ${llama_SOURCE_DIR}/tools/mtmd
        ${llama_SOURCE_DIR}/vendor
)
target_link_libraries(vlm_core PUBLIC llama ggml mtmd llama-common)

# protobuf + ZeroMQ are needed by the two RPC servers and by nothing else -- not
# by vla_core, the CLI, the bench harness or the tests. On a cluster node with no
# package manager and no root those three dependencies are unobtainable, and an
# unconditional REQUIRED aborts the configure before any of the code that does
# build is reached. Default ON, so an ordinary checkout is unaffected.
option(VLA_BUILD_SERVERS "Build vla-server and vlm-server (needs protobuf, libzmq, cppzmq)" ON)

if(VLA_BUILD_SERVERS)
find_package(Protobuf REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(ZeroMQ REQUIRED IMPORTED_TARGET libzmq)

find_path(CPPZMQ_INCLUDE_DIR NAMES zmq.hpp)
if(NOT CPPZMQ_INCLUDE_DIR)
    message(FATAL_ERROR
        "cppzmq (zmq.hpp) not found. Install 'cppzmq-dev' (Debian/Ubuntu) or set "
        "CMAKE_PREFIX_PATH to your cppzmq install prefix. "
        "-DVLA_BUILD_SERVERS=OFF builds everything except the two servers.")
endif()

set(VLA_PROTO_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/serving/vla.proto)
set(VLA_PROTO_GEN_DIR ${CMAKE_CURRENT_BINARY_DIR}/proto-gen/serving)
file(MAKE_DIRECTORY ${VLA_PROTO_GEN_DIR})

add_custom_command(
    OUTPUT  ${VLA_PROTO_GEN_DIR}/vla.pb.h
            ${VLA_PROTO_GEN_DIR}/vla.pb.cc
    COMMAND ${Protobuf_PROTOC_EXECUTABLE}
            --proto_path=${CMAKE_CURRENT_SOURCE_DIR}/src/serving
            --cpp_out=${VLA_PROTO_GEN_DIR}
            ${VLA_PROTO_SRC}
    DEPENDS ${VLA_PROTO_SRC}
    VERBATIM
)

add_executable(vla-server
    src/serving/server.cpp
    ${VLA_PROTO_GEN_DIR}/vla.pb.cc
)
target_include_directories(vla-server PRIVATE
    ${CMAKE_CURRENT_BINARY_DIR}/proto-gen
    ${llama_SOURCE_DIR}/vendor/stb
    ${CPPZMQ_INCLUDE_DIR}
)
target_link_libraries(vla-server PRIVATE
    vla_core
    protobuf::libprotobuf
    PkgConfig::ZeroMQ
)

set(VLM_PROTO_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/serving/vlm.proto)
set(VLM_PROTO_GEN_DIR ${CMAKE_CURRENT_BINARY_DIR}/proto-gen/serving)
file(MAKE_DIRECTORY ${VLM_PROTO_GEN_DIR})

add_custom_command(
    OUTPUT  ${VLM_PROTO_GEN_DIR}/vlm.pb.h
            ${VLM_PROTO_GEN_DIR}/vlm.pb.cc
    COMMAND ${Protobuf_PROTOC_EXECUTABLE}
            --proto_path=${CMAKE_CURRENT_SOURCE_DIR}/src/serving
            --cpp_out=${VLM_PROTO_GEN_DIR}
            ${VLM_PROTO_SRC}
    DEPENDS ${VLM_PROTO_SRC}
    VERBATIM
)

add_executable(vlm-server
    src/serving/vlm-server.cpp
    ${VLM_PROTO_GEN_DIR}/vlm.pb.cc
)
target_include_directories(vlm-server PRIVATE
    ${CMAKE_CURRENT_BINARY_DIR}/proto-gen
    ${llama_SOURCE_DIR}/vendor/stb
    ${CPPZMQ_INCLUDE_DIR}
)
target_link_libraries(vlm-server PRIVATE
    vlm_core
    protobuf::libprotobuf
    PkgConfig::ZeroMQ
)
endif()  # VLA_BUILD_SERVERS

add_library(vla SHARED src/vla_c_api.cpp)
target_include_directories(vla
    PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(vla PRIVATE vla_core)
set_target_properties(vla PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
    PUBLIC_HEADER ${CMAKE_CURRENT_SOURCE_DIR}/include/vla.h
)

add_executable(vla-cli
    src/serving/vla-cli.cpp
)
target_include_directories(vla-cli PRIVATE
    ${llama_SOURCE_DIR}/vendor/stb
)
target_link_libraries(vla-cli PRIVATE vla_core)
# --text shells out to the tokenizer script; VLA_TOKENIZE_SCRIPT overrides it.
target_compile_definitions(vla-cli PRIVATE VLA_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

add_executable(vla-bench
    src/serving/vla-bench.cpp
)
target_link_libraries(vla-bench PRIVATE vla_core)

# Warnings and LTO for our own targets only, never the llama.cpp subtree.
set(VLA_FIRST_PARTY_TARGETS vla_core vlm_core vla vla-cli vla-bench)
if(VLA_BUILD_SERVERS)
    list(APPEND VLA_FIRST_PARTY_TARGETS vla-server vlm-server)
endif()
# The CUDA targets were the only first-party code compiled without warnings.
if(GGML_CUDA)
    list(APPEND VLA_FIRST_PARTY_TARGETS bitvla_cuda_kernels vla_cuda_ops)
endif()

foreach(tgt IN LISTS VLA_FIRST_PARTY_TARGETS)
    target_compile_options(${tgt} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra>)
endforeach()

include(CheckIPOSupported)
check_ipo_supported(RESULT VLA_IPO_OK OUTPUT VLA_IPO_MSG)
if(VLA_IPO_OK AND CMAKE_BUILD_TYPE STREQUAL "Release")
    set(VLA_IPO_TARGETS ${VLA_FIRST_PARTY_TARGETS})
    if(GGML_CUDA)
        # Not the CUDA libraries. IPO turns the device link into an nvlink -dlto,
        # and nvlink refuses to LTO translation units that disagree about -ftz -
        # which these two deliberately do: bitvla_cuda_kernels is built with
        # --use_fast_math (the BitVLA kernels were written against it and the
        # precision contract is stated relative to that), vla_cuda_ops is not,
        # because it feeds ggml ops that must not flush denormals. The flag
        # difference is the point, so the LTO has to go rather than the flag.
        list(REMOVE_ITEM VLA_IPO_TARGETS bitvla_cuda_kernels vla_cuda_ops)
    endif()
    foreach(tgt IN LISTS VLA_IPO_TARGETS)
        set_property(TARGET ${tgt} PROPERTY INTERPROCEDURAL_OPTIMIZATION ON)
    endforeach()
else()
    message(STATUS "vla: IPO/LTO disabled (${VLA_IPO_MSG})")
endif()

option(VLA_BUILD_TESTS "Build vla.cpp test harnesses and unit tests" OFF)
if(VLA_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
