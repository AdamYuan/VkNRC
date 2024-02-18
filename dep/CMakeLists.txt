cmake_minimum_required(VERSION 3.15)
project(SE3306_Final)

add_subdirectory(MyVK)
add_subdirectory(glm)
add_subdirectory(stb_image)
add_subdirectory(tinyobjloader)
add_subdirectory(spdlog)

add_library(dep INTERFACE)
target_link_libraries(dep INTERFACE
        myvk::vulkan myvk::glfw myvk::rg myvk::imgui
        glm
        stb_image
        tinyobjloader
        spdlog::spdlog
)