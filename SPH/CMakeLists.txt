cmake_minimum_required(VERSION 3.22)

project(SPH LANGUAGES CXX CUDA)

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
set(SPH_CUDA_ARCH "native" CACHE STRING "CUDA arch: 'native' or a number like 52;75;86")
option(SPH_ENABLE_RDC "Enable CUDA separable compilation (RDC)" ON)
option(SPH_ENABLE_PIC "Build position independent code (PIC)" ON)

# ============================================================
# CUDA arch
#   cmake -DSPH_CUDA_ARCH=75 ..
#   cmake -DSPH_CUDA_ARCH=52 ..
#   cmake -DSPH_CUDA_ARCH="75;86" ..
#   cmake -DSPH_CUDA_ARCH=native ..
# ============================================================
if (SPH_CUDA_ARCH STREQUAL "native")
    set(CMAKE_CUDA_ARCHITECTURES native)
else()
    # If user passes a single number or a semicolon list, this works.
    set(CMAKE_CUDA_ARCHITECTURES ${SPH_CUDA_ARCH})
endif()
message(STATUS "CMAKE_CUDA_ARCHITECTURES = ${CMAKE_CUDA_ARCHITECTURES}")

# ============================================================
# Paths
# ============================================================
set(SPH_ROOT   ${CMAKE_CURRENT_SOURCE_DIR})
set(KERNEL_DIR ${SPH_ROOT}/kernel)
set(UTIL_DIR   ${KERNEL_DIR}/myUtility)

# ============================================================
# Helper: apply CUDA / PIC properties per-target
# ============================================================
function(sph_apply_target_props tgt)
    set_target_properties(${tgt} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ${SPH_ENABLE_RDC}
        POSITION_INDEPENDENT_CODE  ${SPH_ENABLE_PIC}
    )
    target_compile_features(${tgt} PUBLIC cxx_std_17)
endfunction()

# ============================================================
# Kernel library (everything under kernel/)
# ============================================================
file(GLOB_RECURSE KERNEL_SRCS CONFIGURE_DEPENDS
    ${KERNEL_DIR}/*.cu
    ${KERNEL_DIR}/*.cpp
    ${KERNEL_DIR}/*.cxx
)

add_library(SPH_kernels STATIC
    ${KERNEL_SRCS}
)

target_include_directories(SPH_kernels PUBLIC
    ${SPH_ROOT}
    ${KERNEL_DIR}
    ${UTIL_DIR}
)

sph_apply_target_props(SPH_kernels)

# If you want explicit cudart linkage (usually not strictly required):
# find_package(CUDAToolkit REQUIRED)
# target_link_libraries(SPH_kernels PUBLIC CUDA::cudart)

# ============================================================
# App sources (top-level only, same level as kernel/)
# - Create one executable per source file that contains main()
# - Other top-level .cpp/.cu are compiled into a common static library and linked
# ============================================================
file(GLOB APP_SRCS CONFIGURE_DEPENDS
    ${SPH_ROOT}/*.cpp
    ${SPH_ROOT}/*.cxx
    ${SPH_ROOT}/*.cu
)

set(APP_MAIN_SRCS "")
set(APP_COMMON_SRCS "")

foreach(src ${APP_SRCS})
    # Heuristic: scan for "main(" in the file (fast, avoids full file read).
    # NOTE: Not a perfect parser, but good enough for typical projects.
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

# Common library for non-main top-level sources (optional)
if (APP_COMMON_SRCS)
    add_library(SPH_app_common STATIC
        ${APP_COMMON_SRCS}
    )

    target_include_directories(SPH_app_common PUBLIC
        ${SPH_ROOT}
        ${KERNEL_DIR}
        ${UTIL_DIR}
    )

    sph_apply_target_props(SPH_app_common)

    target_link_libraries(SPH_app_common PUBLIC
        SPH_kernels
    )
endif()

# Create executables for every top-level file that has main()
if (NOT APP_MAIN_SRCS)
    message(WARNING "No top-level .cpp/.cu containing main() was found under ${SPH_ROOT}")
endif()

foreach(main_src ${APP_MAIN_SRCS})
    get_filename_component(exe_name "${main_src}" NAME_WE)

    add_executable(${exe_name}
        ${main_src}
    )

    target_include_directories(${exe_name} PRIVATE
        ${SPH_ROOT}
        ${KERNEL_DIR}
        ${UTIL_DIR}
    )

    target_link_libraries(${exe_name} PRIVATE
        SPH_kernels
    )

    if (TARGET SPH_app_common)
        target_link_libraries(${exe_name} PRIVATE SPH_app_common)
    endif()

    sph_apply_target_props(${exe_name})
endforeach()