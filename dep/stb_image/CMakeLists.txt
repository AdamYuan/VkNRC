cmake_minimum_required(VERSION 3.15)
project(stb_image)

add_library(stb_image STATIC
        src/stb_image.cpp
)
target_include_directories(stb_image PUBLIC include)
target_compile_definitions(stb_image PUBLIC
        STBI_NO_STDIO
        # STBI_NO_PNG
        # STBI_NO_JPEG
        STBI_NO_BMP
        STBI_NO_PSD
        STBI_NO_TGA
        STBI_NO_GIF
        STBI_NO_HDR
        STBI_NO_PIC
        STBI_NO_PNM
        STBI_NO_FAILURE_STRINGS
)
