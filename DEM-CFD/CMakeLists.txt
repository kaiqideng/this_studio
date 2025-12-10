cmake_minimum_required(VERSION 3.22)

set(CMAKE_C_COMPILER /usr/bin/gcc-12)
set(CMAKE_CXX_COMPILER /usr/bin/g++-12)
set(CMAKE_CUDA_HOST_COMPILER /usr/bin/g++-12)
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -G -O0")

project(DEM_CFD LANGUAGES CXX CUDA)

# ===== Language standards =====
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Default build type
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Build type" FORCE)
endif()

# ===== Core sources under kernel/ (should NOT contain any main()) =====
set(SOURCE_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/kernel)

file(GLOB_RECURSE CORE_CPP   CONFIGURE_DEPENDS "${SOURCE_ROOT}/*.cpp")
file(GLOB_RECURSE CORE_CUDA  CONFIGURE_DEPENDS "${SOURCE_ROOT}/*.cu")

add_library(demcfd_core STATIC
    ${CORE_CPP}
    ${CORE_CUDA}
)

target_include_directories(demcfd_core PUBLIC
    ${SOURCE_ROOT}
    ${SOURCE_ROOT}/myContainer
    ${SOURCE_ROOT}/myContainer/myUtility
)

# Enable separable compilation for multiple .cu files
set_target_properties(demcfd_core PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Set CUDA architectures if not provided from the command line
if (NOT CMAKE_CUDA_ARCHITECTURES)
    # Adjust to your GPU: e.g. 75 (Turing), 86 (RTX30), 89 (RTX40), etc.
    set_property(TARGET demcfd_core PROPERTY CUDA_ARCHITECTURES 86)
endif()

# Enable -G for CUDA in Debug builds (better debugging, slower kernels)
target_compile_options(demcfd_core PRIVATE
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:-G>
)

# Output directory for the static library
set_target_properties(demcfd_core PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)

# ===== Executables: one per .cpp in the project root (each with its own main) =====
file(GLOB ROOT_MAINS CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

foreach(main_src IN LISTS ROOT_MAINS)
    # Use the source file name (without extension) as the executable name
    # e.g. problem.cpp -> problem
    get_filename_component(exe_name "${main_src}" NAME_WE)

    add_executable(${exe_name} ${main_src})
    target_link_libraries(${exe_name} PRIVATE demcfd_core)

    # Put all executables into build/bin
    set_target_properties(${exe_name} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
endforeach()