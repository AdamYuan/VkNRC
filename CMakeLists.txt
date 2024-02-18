cmake_minimum_required(VERSION 3.27)
project(VkNRC)

set(CMAKE_CXX_STANDARD 20)

add_subdirectory(dep)
add_subdirectory(shader)

add_executable(VkNRC src/main.cpp)
target_link_libraries(VkNRC PRIVATE dep shader)
