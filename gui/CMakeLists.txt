# =============================================================================
# gui/CMakeLists.txt — interactive ImGui-based simulation viewer.
#
# Pulls in ImGui and GLFW via FetchContent so the build is self-contained.
# Links OpenGL for rendering through ImGui's opengl3 backend.
# =============================================================================

include(FetchContent)

# ---- GLFW (windowing + OpenGL context) -------------------------------------
FetchContent_Declare(
    glfw
    GIT_REPOSITORY https://github.com/glfw/glfw.git
    GIT_TAG        3.4
)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_DOCS     OFF CACHE BOOL "" FORCE)
set(GLFW_INSTALL        OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(glfw)

# ---- Dear ImGui ------------------------------------------------------------
FetchContent_Declare(
    imgui
    GIT_REPOSITORY https://github.com/ocornut/imgui.git
    GIT_TAG        v1.91.5
)
FetchContent_MakeAvailable(imgui)

# Build ImGui as a static library bundling the GLFW + OpenGL3 backends.
add_library(imgui_lib STATIC
    ${imgui_SOURCE_DIR}/imgui.cpp
    ${imgui_SOURCE_DIR}/imgui_demo.cpp
    ${imgui_SOURCE_DIR}/imgui_draw.cpp
    ${imgui_SOURCE_DIR}/imgui_tables.cpp
    ${imgui_SOURCE_DIR}/imgui_widgets.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp
)
target_include_directories(imgui_lib PUBLIC
    ${imgui_SOURCE_DIR}
    ${imgui_SOURCE_DIR}/backends
)
target_link_libraries(imgui_lib PUBLIC glfw)

# ---- OpenGL ----------------------------------------------------------------
find_package(OpenGL REQUIRED)

# ---- The GUI executable ----------------------------------------------------
add_executable(sim_gui SimulationGUI.cpp)
target_link_libraries(sim_gui PRIVATE
    brownian_lib
    imgui_lib
    glfw
    OpenGL::GL
)
target_compile_options(sim_gui PRIVATE -Wall -Wextra -Wpedantic)

# Silence macOS OpenGL deprecation warnings (system OpenGL still works).
if(APPLE)
    target_compile_definitions(sim_gui PRIVATE GL_SILENCE_DEPRECATION)
endif()
