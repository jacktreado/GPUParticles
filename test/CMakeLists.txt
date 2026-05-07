# =============================================================================
# test/CMakeLists.txt
# Unit and integration tests for the 2D Brownian dynamics engine.
# Each module gets its own source file; all tests share one executable so
# the build stays fast. Run with: ctest --test-dir build -V
# =============================================================================

set(TEST_SOURCES
    test_box.cpp
    test_system.cpp
    test_force_calculator.cpp
    test_cell_list.cpp
    test_integrator.cpp
    test_heun_integrator.cpp
    test_adaptive_integrator.cpp
    test_adaptive_macro_integrator.cpp
    test_active_integrator.cpp
    test_config.cpp
    test_integration.cpp
)

add_executable(bd_tests ${TEST_SOURCES})

target_link_libraries(bd_tests PRIVATE
    brownian_lib
    GTest::gtest_main
)

target_compile_options(bd_tests PRIVATE -Wall -Wextra -Wpedantic)

include(GoogleTest)
gtest_discover_tests(bd_tests)
