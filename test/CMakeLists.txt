cmake_minimum_required(VERSION 3.27)
project(VkNRC)

find_package(Vulkan REQUIRED)
add_executable(VkNRC_NN_Test main.cpp)
target_include_directories(VkNRC_NN_Test PRIVATE
        vuda
        half
        eigen
)
target_link_libraries(VkNRC_NN_Test PRIVATE Vulkan::Vulkan)

