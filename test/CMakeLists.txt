cmake_minimum_required(VERSION 3.27)
project(VkNRC)

set(CMAKE_CXX_STANDARD 20)
find_package(Vulkan REQUIRED)
add_executable(VkNRC_NN_Test main.cpp)
target_include_directories(VkNRC_NN_Test PRIVATE
        vuda
        half
        eigen
)
target_link_libraries(VkNRC_NN_Test PRIVATE Vulkan::Vulkan)

add_subdirectory(../dep/MyVK dep/MyVK)
add_subdirectory(../dep/stb_image dep/stb_image)
add_subdirectory(../dep/half dep/half)

add_executable(MLP_Learning_An_Image mlp_learning_an_image/main.cpp)
target_link_libraries(MLP_Learning_An_Image PRIVATE myvk::vulkan myvk::glfw myvk::rg myvk::imgui stb_image half::half)

