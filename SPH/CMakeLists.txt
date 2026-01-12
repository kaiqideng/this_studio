cmake_minimum_required(VERSION 3.22)

# ============================================================
# Export compile_commands.json for clangd
# ============================================================
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ============================================================
# Optional: try gcc-12/g++-12 (Linux/WSL)
# NOTE: Compiler must be set BEFORE project()
# ============================================================
option(SPH_TRY_GCC12 "Use gcc-12/g++-12 if available" ON)
if (SPH_TRY_GCC12)
    find_program(SPH_GCC gcc-12)
    find_program(SPH_GXX g++-12)
    if (SPH_GCC AND SPH_GXX)
        set(CMAKE_C_COMPILER         "${SPH_GCC}")
        set(CMAKE_CXX_COMPILER       "${SPH_GXX}")
        set(CMAKE_CUDA_HOST_COMPILER "${SPH_GXX}")
        message(STATUS "Using gcc-12/g++-12: ${SPH_GCC}, ${SPH_GXX}")
    endif()
endif()

project(SPH LANGUAGES CXX CUDA)

# ============================================================
# Standards
# ============================================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# ============================================================
# Build type (single-config generators)
# ============================================================
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "" FORCE)
endif()

# ============================================================
# CUDA architectures
# ============================================================
option(SPH_CUDA_USE_NATIVE_ARCH "Use CMAKE_CUDA_ARCHITECTURES=native" ON)
if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    if (SPH_CUDA_USE_NATIVE_ARCH)
        set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "CUDA arch list" FORCE)
    else()
        set(CMAKE_CUDA_ARCHITECTURES 52 CACHE STRING "CUDA arch list" FORCE)
    endif()
endif()
message(STATUS "CMAKE_CUDA_ARCHITECTURES = ${CMAKE_CUDA_ARCHITECTURES}")

# ============================================================
# Paths: ProjectRoot/DEM and ProjectRoot/SPH
# ============================================================
get_filename_component(PROJECT_ROOT "${CMAKE_CURRENT_LIST_DIR}/.." REALPATH)
set(DEM_ROOT     "${PROJECT_ROOT}/DEM")
set(DEM_KERNEL   "${DEM_ROOT}/kernel")
set(SPH_ROOT     "${CMAKE_CURRENT_LIST_DIR}")
set(SPH_KERNEL   "${SPH_ROOT}/kernel")

if (NOT EXISTS "${DEM_KERNEL}")
    message(FATAL_ERROR "Cannot find DEM kernel at: ${DEM_KERNEL}")
endif()
if (NOT EXISTS "${SPH_KERNEL}")
    message(FATAL_ERROR "Cannot find SPH kernel at: ${SPH_KERNEL}")
endif()

message(STATUS "PROJECT_ROOT = ${PROJECT_ROOT}")
message(STATUS "DEM_ROOT     = ${DEM_ROOT}")
message(STATUS "SPH_ROOT     = ${SPH_ROOT}")

# ============================================================
# Common CUDA compile options (debug-friendly)
# ============================================================
add_library(cuda_flags INTERFACE)
target_compile_options(cuda_flags INTERFACE
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
)

# ============================================================
# DEM: build inside SPH project
# ============================================================
file(GLOB_RECURSE DEM_CORE_CPP  CONFIGURE_DEPENDS "${DEM_KERNEL}/*.cpp")
file(GLOB_RECURSE DEM_CORE_CUDA CONFIGURE_DEPENDS "${DEM_KERNEL}/*.cu")

# TU that calls thrust::sort_by_key -> RDC OFF
set(DEM_SORT_CUDA "${DEM_KERNEL}/cudaKernel/cudaSort/biuldHashStartEnd.cu")
list(FILTER DEM_CORE_CUDA EXCLUDE REGEX ".*[/\\\\]biuldHashStartEnd\\.cu$")

add_library(DEM_sort STATIC ${DEM_SORT_CUDA})
target_include_directories(DEM_sort PUBLIC
    ${DEM_KERNEL}
    ${DEM_KERNEL}/myStruct
    ${DEM_KERNEL}/myStruct/myUtility
)
set_target_properties(DEM_sort PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)
target_link_libraries(DEM_sort PRIVATE cuda_flags)

add_library(DEM_core STATIC ${DEM_CORE_CPP} ${DEM_CORE_CUDA})
target_include_directories(DEM_core PUBLIC
    ${DEM_KERNEL}
    ${DEM_KERNEL}/myStruct
    ${DEM_KERNEL}/myStruct/myUtility
)
set_target_properties(DEM_core PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)
target_link_libraries(DEM_core PUBLIC DEM_sort)
target_link_libraries(DEM_core PRIVATE cuda_flags)

# ============================================================
# SPH core: everything under SPH/kernel
# ============================================================
file(GLOB_RECURSE SPH_CPP CONFIGURE_DEPENDS "${SPH_KERNEL}/*.cpp")
file(GLOB_RECURSE SPH_CU  CONFIGURE_DEPENDS "${SPH_KERNEL}/*.cu")
file(GLOB_RECURSE SPH_H   CONFIGURE_DEPENDS "${SPH_KERNEL}/*.h")

add_library(SPH_core STATIC ${SPH_CPP} ${SPH_CU} ${SPH_H})

# IMPORTANT:
# - ${PROJECT_ROOT} enables: #include "DEM/kernel/xxx.h"
# - ${DEM_KERNEL} etc enables: #include "cudaKernel/..." or "myStruct/..."
target_include_directories(SPH_core PUBLIC
    ${SPH_KERNEL}
    ${PROJECT_ROOT}
    ${DEM_KERNEL}
    ${DEM_KERNEL}/myStruct
    ${DEM_KERNEL}/myStruct/myUtility
)

set_target_properties(SPH_core PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)

target_link_libraries(SPH_core PUBLIC DEM_core)
target_link_libraries(SPH_core PRIVATE cuda_flags)

# ============================================================
# Executables: all *.cpp directly under SPH/ (e.g. damBreak.cpp)
# ============================================================
file(GLOB SPH_MAINS CONFIGURE_DEPENDS "${SPH_ROOT}/*.cpp")

foreach(main_src IN LISTS SPH_MAINS)
    get_filename_component(exe_name "${main_src}" NAME_WE)
    add_executable(${exe_name} ${main_src})
    target_link_libraries(${exe_name} PRIVATE SPH_core)
    set_target_properties(${exe_name} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
endforeach()