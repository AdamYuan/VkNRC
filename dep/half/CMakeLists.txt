cmake_minimum_required(VERSION 3.15)
project(half)

add_library(half INTERFACE)
add_library(half::half ALIAS half)
target_include_directories(half INTERFACE include)
