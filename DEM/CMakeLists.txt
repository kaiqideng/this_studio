cmake_minimum_required(VERSION 3.22)

# ============================================================
# Compilers (must be set BEFORE project())
# ============================================================
set(CMAKE_C_COMPILER         /usr/bin/gcc-12)
set(CMAKE_CXX_COMPILER       /usr/bin/g++-12)
set(CMAKE_CUDA_HOST_COMPILER /usr/bin/g++-12)

project(DEM LANGUAGES CXX CUDA)

# ============================================================
# Language standards
# ============================================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# ============================================================
# Build type
# ============================================================
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "" FORCE)
endif()

# ============================================================
# CUDA architectures (AUTO)
# ============================================================
option(DEM_CUDA_USE_NATIVE_ARCH
       "Auto-detect GPU arch on this machine (CMAKE_CUDA_ARCHITECTURES=native)"
       ON)

if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    if (DEM_CUDA_USE_NATIVE_ARCH)
        # Best when configuring on a machine that has a GPU
        set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "CUDA arch list" FORCE)
    else()
        # Portable fallback: adjust if you want fewer/more archs
        set(CMAKE_CUDA_ARCHITECTURES 52;60;61;70;75;80;86;89 CACHE STRING "CUDA arch list" FORCE)
    endif()
endif()

message(STATUS "CMAKE_CUDA_ARCHITECTURES = ${CMAKE_CUDA_ARCHITECTURES}")

# ============================================================
# Sources
# ============================================================
set(SOURCE_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/kernel)

file(GLOB_RECURSE CORE_CPP  CONFIGURE_DEPENDS "${SOURCE_ROOT}/*.cpp")
file(GLOB_RECURSE CORE_CUDA CONFIGURE_DEPENDS "${SOURCE_ROOT}/*.cu")

# ---- The TU that calls thrust::sort_by_key (NO RDC) ----
set(SORT_CUDA "${SOURCE_ROOT}/cudaKernel/cudaSort/biuldHashStartEnd.cu")

# Remove SORT_CUDA from core list by regex (most compatible)
# This matches the file name regardless of absolute/relative path style.
list(FILTER CORE_CUDA EXCLUDE REGEX ".*/biuldHashStartEnd\\.cu$")

# ============================================================
# DEM_sort (RDC OFF)
# ============================================================
add_library(DEM_sort STATIC
    ${SORT_CUDA}
)

target_include_directories(DEM_sort PUBLIC
    ${SOURCE_ROOT}
    ${SOURCE_ROOT}/myStruct
    ${SOURCE_ROOT}/myStruct/myUtility
)

set_target_properties(DEM_sort PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
)

# ============================================================
# DEM_core (RDC ON)
# ============================================================
add_library(DEM_core STATIC
    ${CORE_CPP}
    ${CORE_CUDA}
)

target_include_directories(DEM_core PUBLIC
    ${SOURCE_ROOT}
    ${SOURCE_ROOT}/myStruct
    ${SOURCE_ROOT}/myStruct/myUtility
)

set_target_properties(DEM_core PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
)

# Core depends on sort
target_link_libraries(DEM_core PUBLIC DEM_sort)

# Optional: CUDA lineinfo (debug-friendly)
target_compile_options(DEM_core PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
target_compile_options(DEM_sort PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)

# ============================================================
# Executables
# ============================================================
file(GLOB ROOT_MAINS CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

foreach(main_src IN LISTS ROOT_MAINS)
    get_filename_component(exe_name "${main_src}" NAME_WE)
    add_executable(${exe_name} ${main_src})
    target_link_libraries(${exe_name} PRIVATE DEM_core)
    set_target_properties(${exe_name} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
endforeach()