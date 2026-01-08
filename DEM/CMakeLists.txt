cmake_minimum_required(VERSION 3.22)

# Compilers should be set before project()
set(CMAKE_C_COMPILER /usr/bin/gcc-12)
set(CMAKE_CXX_COMPILER /usr/bin/g++-12)
set(CMAKE_CUDA_HOST_COMPILER /usr/bin/g++-12)

project(DEM LANGUAGES CXX CUDA)

# ===== Language standards =====
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Default build type
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Build type" FORCE)
endif()

# ===== CUDA architectures (UNIFY HERE) =====
if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75 CACHE STRING "CUDA arch list" FORCE)
endif()
message(STATUS "CMAKE_CUDA_ARCHITECTURES = ${CMAKE_CUDA_ARCHITECTURES}")

# ===== Sources =====
set(SOURCE_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/kernel)

file(GLOB_RECURSE CORE_CPP  CONFIGURE_DEPENDS "${SOURCE_ROOT}/*.cpp")
file(GLOB_RECURSE CORE_CUDA CONFIGURE_DEPENDS "${SOURCE_ROOT}/*.cu")

set(SORT_CUDA "${SOURCE_ROOT}/cudaKernel/biuldHashStartEnd.cu")
list(REMOVE_ITEM CORE_CUDA "${SORT_CUDA}")

# ===== DEM_sort (RDC OFF) =====
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
)

set_property(TARGET DEM_sort PROPERTY CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES})

# ===== DEM_core (RDC ON) =====
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
)

set_property(TARGET DEM_core PROPERTY CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES})

# Core depends on sort (so executables only need to link DEM_core)
target_link_libraries(DEM_core PUBLIC DEM_sort)

# Optional: debug CUDA flags (better as target options, but OK globally too)
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -O0 -lineinfo")

# ===== Executables =====
file(GLOB ROOT_MAINS CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

foreach(main_src IN LISTS ROOT_MAINS)
    get_filename_component(exe_name "${main_src}" NAME_WE)
    add_executable(${exe_name} ${main_src})
    target_link_libraries(${exe_name} PRIVATE DEM_core)
    set_target_properties(${exe_name} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
endforeach()