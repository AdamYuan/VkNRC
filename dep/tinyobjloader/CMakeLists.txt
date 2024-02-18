cmake_minimum_required(VERSION 3.15)
project(tinyobjloader)

add_library(tinyobjloader STATIC
        src/tiny_obj_loader.cc
)
target_include_directories(tinyobjloader PUBLIC include)
