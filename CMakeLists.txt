cmake_minimum_required(VERSION 3.16)
project(BrownianSim CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------
# Boost.Odeint is header-only; we only need the include path.
# Install via: brew install boost
find_package(Boost REQUIRED)

# HDF5 C++ library for trajectory output.
# Install via: brew install hdf5
find_package(HDF5 REQUIRED COMPONENTS CXX)

# Try a system-installed nlohmann_json first, fall back to FetchContent.
find_package(nlohmann_json 3.10 QUIET)
if(NOT nlohmann_json_FOUND)
    include(FetchContent)
    FetchContent_Declare(
        nlohmann_json
        GIT_REPOSITORY https://github.com/nlohmann/json.git
        GIT_TAG        v3.11.3
    )
    FetchContent_MakeAvailable(nlohmann_json)
endif()

# ---------------------------------------------------------------------------
# Library: all the simulation engine code lives here. Keeping it as a
# library (separate from main.cpp) makes it trivial to add unit tests or
# alternative drivers later.
# ---------------------------------------------------------------------------
add_library(brownian_lib
    src/System.cpp
    src/ForceCalculator.cpp
    src/CellList.cpp
    src/Integrator.cpp
    src/HeunIntegrator.cpp
    src/AdaptiveIntegrator.cpp
    src/AdaptiveMacroIntegrator.cpp
    src/RandomGenerator.cpp
    src/Initializer.cpp
    src/Config.cpp
    src/TrajectoryWriter.cpp
    src/CorrelationAccumulator.cpp
    src/ContactDurationAccumulator.cpp
)

target_include_directories(brownian_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(brownian_lib PUBLIC
    nlohmann_json::nlohmann_json
    Boost::headers
    ${HDF5_CXX_LIBRARIES}
)
target_include_directories(brownian_lib PUBLIC ${HDF5_INCLUDE_DIRS})

target_compile_options(brownian_lib PRIVATE
    -Wall -Wextra -Wpedantic -O3
)

# ---------------------------------------------------------------------------
# Executable
# ---------------------------------------------------------------------------
add_executable(sim main.cpp)
target_link_libraries(sim PRIVATE brownian_lib)
target_compile_options(sim PRIVATE -Wall -Wextra -Wpedantic -O3)

# ---------------------------------------------------------------------------
# Tests (Google Test)
# ---------------------------------------------------------------------------
include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.15.2
)
# Prevent installing GTest alongside our project.
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()
add_subdirectory(test)

# ---------------------------------------------------------------------------
# Interactive GUI (ImGui + GLFW + OpenGL).  Optional — disable with -DBUILD_GUI=OFF.
# ---------------------------------------------------------------------------
option(BUILD_GUI "Build the interactive ImGui-based simulation GUI" OFF)
if(BUILD_GUI)
    add_subdirectory(gui)
endif()
