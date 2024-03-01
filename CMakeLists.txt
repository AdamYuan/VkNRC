cmake_minimum_required(VERSION 3.27)
project(VkNRC)

set(CMAKE_CXX_STANDARD 20)

set(CMAKE_CXX_VISIBILITY_PRESET hidden)

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

add_subdirectory(dep)
add_subdirectory(shader)
add_subdirectory(test)

add_executable(VkNRC
        src/main.cpp
        src/Camera.cpp
        src/Scene.cpp
        # src/Sobol.cpp
        # src/BlueNoise.cpp
        src/VkScene.cpp
        src/VkSceneBLAS.cpp
        src/VkSceneTLAS.cpp
        src/VkNRCState.cpp

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
