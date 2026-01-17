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
# CUDA arch (portable default, can override)
#   cmake -DSPH_CUDA_ARCH=75 ..
# ============================================================
if (SPH_CUDA_ARCH STREQUAL "native")
    set(CMAKE_CUDA_ARCHITECTURES native)
else()
    set(CMAKE_CUDA_ARCHITECTURES ${SPH_CUDA_ARCH})
endif()
message(STATUS "CMAKE_CUDA_ARCHITECTURES = ${CMAKE_CUDA_ARCHITECTURES}")

# ============================================================
# RDC / separable compilation
# ============================================================
option(SPH_ENABLE_RDC "Enable CUDA separable compilation (RDC)" ON)
if (SPH_ENABLE_RDC)
    set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
endif()

# ============================================================
# Paths
# ============================================================
set(SPH_ROOT ${CMAKE_CURRENT_SOURCE_DIR})
set(KERNEL_DIR ${SPH_ROOT}/kernel)
set(UTIL_DIR   ${KERNEL_DIR}/myUtility)

# ============================================================
# Kernel library (everything under kernel/)
# ============================================================
file(GLOB KERNEL_CU CONFIGURE_DEPENDS
    ${KERNEL_DIR}/*.cu
)
file(GLOB KERNEL_CPP CONFIGURE_DEPENDS
    ${KERNEL_DIR}/*.cpp
)
file(GLOB KERNEL_HEADERS CONFIGURE_DEPENDS
    ${KERNEL_DIR}/*.h
    ${KERNEL_DIR}/*.hpp
    ${UTIL_DIR}/*.h
    ${UTIL_DIR}/*.hpp
)

add_library(SPH_kernels STATIC
    ${KERNEL_CU}
    ${KERNEL_CPP}
    ${KERNEL_HEADERS}
)

target_include_directories(SPH_kernels PUBLIC
    ${SPH_ROOT}
    ${KERNEL_DIR}
    ${UTIL_DIR}
)

target_compile_features(SPH_kernels PUBLIC cxx_std_17)

set_target_properties(SPH_kernels PROPERTIES
    CUDA_SEPARABLE_COMPILATION ${SPH_ENABLE_RDC}
    POSITION_INDEPENDENT_CODE ON
)

# ============================================================
# App sources (top-level only, same level as kernel/)
# - Create one executable per source file that contains main()
# - Other top-level .cpp/.cu are compiled into a common static library and linked
# ============================================================
file(GLOB APP_CPP CONFIGURE_DEPENDS
    ${SPH_ROOT}/*.cpp
)
file(GLOB APP_CU CONFIGURE_DEPENDS
    ${SPH_ROOT}/*.cu
)
set(APP_SRCS ${APP_CPP} ${APP_CU})

set(APP_MAIN_SRCS "")
set(APP_COMMON_SRCS "")

foreach(src ${APP_SRCS})
    file(READ "${src}" _src_text)
    string(REGEX MATCH "([\\n\\r\\t ]|^)main[\\t ]*\\(" _has_main "${_src_text}")
    if (_has_main)
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

    target_compile_features(SPH_app_common PUBLIC cxx_std_17)

    set_target_properties(SPH_app_common PROPERTIES
        CUDA_SEPARABLE_COMPILATION ${SPH_ENABLE_RDC}
        POSITION_INDEPENDENT_CODE ON
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

    target_compile_features(${exe_name} PRIVATE cxx_std_17)

    set_target_properties(${exe_name} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ${SPH_ENABLE_RDC}
    )
endforeach()