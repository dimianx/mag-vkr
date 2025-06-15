cmake_minimum_required(VERSION 3.14)
project(vkr VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(VKR_BUILD_TESTS "Build unit tests" ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
endif()

include(FetchContent)
set(FETCHCONTENT_QUIET OFF)

FetchContent_Declare(
  eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG        3.4.0
  GIT_SHALLOW    TRUE
  GIT_PROGRESS   TRUE
)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(eigen)

FetchContent_Declare(
  xsimd
  GIT_REPOSITORY https://github.com/xtensor-stack/xsimd.git
  GIT_TAG        11.1.0
  GIT_SHALLOW    TRUE
  GIT_PROGRESS   TRUE
)
set(BUILD_TESTS OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(xsimd)

FetchContent_Declare(
  mio
  GIT_REPOSITORY https://github.com/vimpunk/mio.git
  GIT_TAG        master
  GIT_SHALLOW    TRUE
  GIT_PROGRESS   TRUE
)
FetchContent_MakeAvailable(mio)

FetchContent_Declare(
  json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG        v3.11.2
  GIT_SHALLOW    TRUE
  GIT_PROGRESS   TRUE
)
set(JSON_BuildTests OFF CACHE BOOL "" FORCE)
set(JSON_Install OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(json)

FetchContent_Declare(
  tbb
  GIT_REPOSITORY https://github.com/oneapi-src/oneTBB.git
  GIT_TAG        v2021.11.0
  GIT_SHALLOW    TRUE
  GIT_PROGRESS   TRUE
)
set(TBB_TEST OFF CACHE BOOL "" FORCE)
set(TBB_STRICT OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(tbb)

FetchContent_Declare(
  spdlog
  GIT_REPOSITORY https://github.com/gabime/spdlog.git
  GIT_TAG        v1.12.0
  GIT_SHALLOW    TRUE
  GIT_PROGRESS   TRUE
)
set(SPDLOG_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(SPDLOG_BUILD_BENCH OFF CACHE BOOL "" FORCE)
set(SPDLOG_BUILD_TESTS OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(spdlog)


find_package(GDAL REQUIRED)

if(VKR_BUILD_TESTS)
  FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.14.0
    GIT_SHALLOW    TRUE
    GIT_PROGRESS   TRUE
  )
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)
endif()

file(GLOB_RECURSE VKR_SOURCES
     CONFIGURE_DEPENDS         
     src/*.cpp
)


add_library(vkr ${VKR_SOURCES})
add_library(vkr::vkr ALIAS vkr)

target_include_directories(vkr
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    src
)

target_link_libraries(vkr
  PUBLIC
    Eigen3::Eigen
    ${GDAL_LIBRARIES}
  PRIVATE
    $<BUILD_INTERFACE:TBB::tbb>
    $<BUILD_INTERFACE:spdlog::spdlog>
)

target_include_directories(vkr
  SYSTEM PUBLIC
    $<BUILD_INTERFACE:${xsimd_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${json_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${tbb_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${spdlog_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${mio_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_include_directories(vkr PUBLIC ${GDAL_INCLUDE_DIRS})

if(VKR_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()


include(GNUInstallDirs)

install(TARGETS vkr
  EXPORT vkrTargets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(DIRECTORY include/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)


message(STATUS "")
message(STATUS "Configuration summary:")
message(STATUS "  Version:          ${PROJECT_VERSION}")
message(STATUS "  Build type:       ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ standard:     ${CMAKE_CXX_STANDARD}")
message(STATUS "  Compiler:         ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "")
message(STATUS "  Build tests:      ${VKR_BUILD_TESTS}")
message(STATUS "")
message(STATUS "  Dependencies:")
message(STATUS "    Eigen3:         ${eigen_SOURCE_DIR}")
message(STATUS "    xsimd:          ${xsimd_SOURCE_DIR}")
message(STATUS "    spdlog:         ${spdlog_SOURCE_DIR}")
message(STATUS "    nlohmann_json:  ${json_SOURCE_DIR}")
message(STATUS "    TBB:            ${tbb_SOURCE_DIR}")
message(STATUS "    GDAL:           ${GDAL_INCLUDE_DIRS}")
message(STATUS "")