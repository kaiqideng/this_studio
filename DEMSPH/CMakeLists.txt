cmake_minimum_required(VERSION 3.22)

project(DEMSPH LANGUAGES CXX CUDA)

# ============================================================
# Standards
# ============================================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ============================================================
# Options
# ============================================================
set(DEMSPH_CUDA_ARCH "native" CACHE STRING "CUDA arch: native or e.g. 75;86")
option(DEMSPH_ENABLE_RDC "Enable CUDA separable compilation (RDC)" ON)
option(DEMSPH_ENABLE_PIC "Build position independent code (PIC)" ON)

# ============================================================
# CUDA arch
# ============================================================
if (DEMSPH_CUDA_ARCH STREQUAL "native")
    set(CMAKE_CUDA_ARCHITECTURES native)
else()
    set(CMAKE_CUDA_ARCHITECTURES ${DEMSPH_CUDA_ARCH})
endif()
message(STATUS "CMAKE_CUDA_ARCHITECTURES = ${CMAKE_CUDA_ARCHITECTURES}")

# ============================================================
# Paths
# ============================================================
set(DEMSPH_ROOT ${CMAKE_CURRENT_SOURCE_DIR})
set(KERNEL_DIR  ${DEMSPH_ROOT}/kernel)
set(UTIL_DIR    ${KERNEL_DIR}/myUtility)

# ============================================================
# Helper: per-target props
# ============================================================
function(demsph_apply tgt)
    set_target_properties(${tgt} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ${DEMSPH_ENABLE_RDC}
        POSITION_INDEPENDENT_CODE  ${DEMSPH_ENABLE_PIC}
    )
    target_compile_features(${tgt} PUBLIC cxx_std_17)
    target_include_directories(${tgt} PUBLIC
        ${DEMSPH_ROOT}
        ${KERNEL_DIR}
        ${UTIL_DIR}
    )
endfunction()

# ============================================================
# Kernel library (everything under kernel/)
# ============================================================
file(GLOB_RECURSE KERNEL_SRCS CONFIGURE_DEPENDS
    ${KERNEL_DIR}/*.cu
    ${KERNEL_DIR}/*.cpp
    ${KERNEL_DIR}/*.cxx
    ${KERNEL_DIR}/*.h
    ${KERNEL_DIR}/*.hpp
)

add_library(DEMSPH_kernels STATIC ${KERNEL_SRCS})
demsph_apply(DEMSPH_kernels)

# ============================================================
# App sources (top-level only)
#   - one exe per main()
#   - other top-level sources -> app_common
# ============================================================
file(GLOB APP_SRCS CONFIGURE_DEPENDS
    ${DEMSPH_ROOT}/*.cpp
    ${DEMSPH_ROOT}/*.cxx
    ${DEMSPH_ROOT}/*.cu
)

set(APP_MAIN_SRCS "")
set(APP_COMMON_SRCS "")

foreach(src ${APP_SRCS})
    file(STRINGS "${src}" _main_hits
        REGEX "[^A-Za-z0-9_]main[ \t\r\n]*\\("
        LIMIT_COUNT 2000
    )

    if (_main_hits)
        list(APPEND APP_MAIN_SRCS "${src}")
    else()
        list(APPEND APP_COMMON_SRCS "${src}")
    endif()
endforeach()

# Common library for non-main top-level sources
if (APP_COMMON_SRCS)
    add_library(DEMSPH_app_common STATIC ${APP_COMMON_SRCS})
    demsph_apply(DEMSPH_app_common)
    target_link_libraries(DEMSPH_app_common PUBLIC DEMSPH_kernels)
endif()

# Create executables for every top-level file that has main()
if (NOT APP_MAIN_SRCS)
    message(FATAL_ERROR "No top-level main() was found under ${DEMSPH_ROOT}")
endif()

foreach(main_src ${APP_MAIN_SRCS})
    get_filename_component(exe_name "${main_src}" NAME_WE)

    add_executable(${exe_name} ${main_src})
    demsph_apply(${exe_name})

    target_link_libraries(${exe_name} PRIVATE DEMSPH_kernels)

    if (TARGET DEMSPH_app_common)
        target_link_libraries(${exe_name} PRIVATE DEMSPH_app_common)
    endif()
endforeach()