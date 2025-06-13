include(GoogleTest)

file(GLOB_RECURSE VKR_UNIT_TEST_SOURCES
     CONFIGURE_DEPENDS
     unit/*.cpp
)

add_executable(vkr_unit_tests ${VKR_UNIT_TEST_SOURCES})
target_link_libraries(vkr_unit_tests
  PRIVATE
    vkr::vkr
    GTest::gtest
    GTest::gtest_main
    
)

gtest_discover_tests(vkr_unit_tests
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
)