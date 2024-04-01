cmake_minimum_required(VERSION 3.8...3.25)

# Fallback for using newer policies on CMake <3.12.
if(${CMAKE_VERSION} VERSION_LESS 3.12)
  cmake_policy(VERSION ${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION})
endif()

add_library(fmt-header-only INTERFACE)
add_library(fmt::fmt-header-only ALIAS fmt-header-only)

target_compile_definitions(fmt-header-only INTERFACE FMT_HEADER_ONLY=1)
target_compile_features(fmt-header-only INTERFACE cxx_std_11)

target_include_directories(fmt-header-only ${FMT_SYSTEM_HEADERS_ATTRIBUTE} INTERFACE
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${FMT_INC_DIR}>)
