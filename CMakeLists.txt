cmake_minimum_required(VERSION 3.30)
project(affinity_propagation)

set(CMAKE_CXX_STANDARD 20)


# Set build type to Release if not specified
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
else ()
    message("Build type: ${CMAKE_BUILD_TYPE}")
endif ()

message(STATUS "Checking OpenMP")
find_package(OpenMP REQUIRED)

add_executable(affinity_propagation src/main.cpp
        src/affinity_propagation.cpp
        src/affinity_propagation.h
        src/mnist_digit.h
        src/consts.h)

if (OpenMP_CXX_FOUND)
    target_link_libraries(affinity_propagation PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(affinity_propagation PRIVATE USE_OPENMP)  # Definice pro překladač
endif ()

# Optimalizace
if (MSVC)
    message(STATUS "Using MSVC")
    target_compile_options(affinity_propagation PRIVATE
            $<$<CONFIG:Release>:/O2 /Ob2 /Oi /Ot /GL> # Optimalizace
            $<$<CONFIG:Debug>:/Od /Zi>                # Debug info, bez optimalizace
    )
    target_link_options(affinity_propagation PRIVATE
            $<$<CONFIG:Release>:/LTCG>                # Link-time code generation
    )

    # OpenMP flagy pro MSVC
    if (OpenMP_CXX_FOUND)
        target_compile_options(affinity_propagation PRIVATE /openmp:llvm) # LLVM backend
    endif ()

elseif (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    message(STATUS "Using GCC or Clang")
    target_compile_options(affinity_propagation PRIVATE
            $<$<CONFIG:Release>:-O3 -march=native -ffast-math>
            $<$<CONFIG:Debug>:-O0 -g>
    )
    # Odkomentuj pro LTO
    # target_link_options(affinity_propagation PRIVATE $<$<CONFIG:Release>:-flto>)  # Enable LTO
    if (OpenMP_CXX_FOUND)
        target_compile_options(affinity_propagation PRIVATE -fopenmp)
        target_link_libraries(affinity_propagation PUBLIC -fopenmp)
    endif ()
endif ()
