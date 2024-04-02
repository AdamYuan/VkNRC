cmake_minimum_required(VERSION 3.27)
project(VkNRC)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)

enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if (CMAKE_SYSTEM_NAME STREQUAL Windows)
    add_definitions(-D_USE_MATH_DEFINES)
    add_definitions(-DNOMINMAX)
    add_definitions(-DWIN32_LEAN_AND_MEAN)
endif ()

if (MINGW)
    set(CMAKE_EXE_LINKER_FLAGS "-static-libgcc -static-libstdc++ -static")
endif ()

if (CMAKE_BUILD_TYPE STREQUAL "Release" AND NOT MINGW)
    include(CheckIPOSupported)
    check_ipo_supported(RESULT IPO_SUPPORTED)
    if (IPO_SUPPORTED)
        message("Enable IPO")
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
    endif ()
endif ()


# Adapted from the CMake source code at https://github.com/Kitware/CMake/blob/master/Modules/FindCUDA/select_compute_arch.cmake
# Simplified to return a semicolon-separated list of the compute capabilities of installed devices
function(AUTODETECT_CUDA_ARCHITECTURES OUT_VARIABLE)
    if (NOT AUTODETECT_CUDA_ARCHITECTURES_OUTPUT)
        if (CMAKE_CUDA_COMPILER_LOADED) # CUDA as a language
            set(file "${PROJECT_BINARY_DIR}/detect_cuda_architectures.cu")
        else ()
            set(file "${PROJECT_BINARY_DIR}/detect_cuda_architectures.cpp")
        endif ()

        file(WRITE ${file} ""
                "#include <cuda_runtime.h>\n"
                "#include <cstdio>\n"
                "int main() {\n"
                "	int count = 0;\n"
                "	if (cudaSuccess != cudaGetDeviceCount(&count)) return -1;\n"
                "	if (count == 0) return -1;\n"
                "	for (int device = 0; device < count; ++device) {\n"
                "		cudaDeviceProp prop;\n"
                "		if (cudaSuccess == cudaGetDeviceProperties(&prop, device)) {\n"
                "			std::printf(\"%d%d\", prop.major, prop.minor);\n"
                "			if (device < count - 1) std::printf(\";\");\n"
                "		}\n"
                "	}\n"
                "	return 0;\n"
                "}\n"
        )

        try_run(run_result compile_result ${PROJECT_BINARY_DIR} ${file} RUN_OUTPUT_VARIABLE compute_capabilities)
        if (run_result EQUAL 0)
            # If the user has multiple GPUs with the same compute capability installed, list that capability only once.
            list(REMOVE_DUPLICATES compute_capabilities)
            set(AUTODETECT_CUDA_ARCHITECTURES_OUTPUT ${compute_capabilities} CACHE INTERNAL "Returned GPU architectures from detect_gpus tool" FORCE)
        endif ()
    endif ()

    if (NOT AUTODETECT_CUDA_ARCHITECTURES_OUTPUT)
        message(STATUS "Automatic GPU detection failed. Building for Turing and Ampere as a best guess.")
        set(${OUT_VARIABLE} "75;86" PARENT_SCOPE)
    else ()
        set(${OUT_VARIABLE} ${AUTODETECT_CUDA_ARCHITECTURES_OUTPUT} PARENT_SCOPE)
    endif ()
endfunction()

AUTODETECT_CUDA_ARCHITECTURES(CMAKE_CUDA_ARCHITECTURES)
set(TCNN_CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES})

add_subdirectory(dep)
add_subdirectory(shader)
# add_subdirectory(test)

add_executable(VkNRC
        src/main.cpp
        src/Camera.cpp
        src/Scene.cpp
        src/VkScene.cpp
        src/VkSceneBLAS.cpp
        src/VkSceneTLAS.cpp
        src/NRCState.cpp
        src/VkNRCResource.cpp
        src/CuVkBuffer.cu
        src/CuNRCNetwork.cu

        src/rg/NRCRenderGraph.cpp
        src/rg/PathTracerPass.cpp
        src/rg/VBufferPass.cpp
        src/rg/ScreenPass.cpp
)
target_link_libraries(VkNRC PRIVATE dep shader)
set_target_properties(VkNRC PROPERTIES
        C_VISIBILITY_PRESET hidden
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN ON
)

install(TARGETS VkNRC RUNTIME DESTINATION)
