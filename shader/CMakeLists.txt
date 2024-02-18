cmake_minimum_required(VERSION 3.15)
project(SE3306_Final)

set(SHADER_HEADER
)

set(SHADER_SRC
)

set(SHADER_INCLUDE_PATH ${CMAKE_CURRENT_LIST_DIR}/include/shader)

list(TRANSFORM SHADER_HEADER PREPEND ${SHADER_INCLUDE_PATH}/)

find_program(GLSLC "glslc" REQUIRED)
# IF (GLSLC)
function(add_spirv_shader INPUT_FILE)
    add_custom_command(OUTPUT ${SHADER_INCLUDE_PATH}/${INPUT_FILE}.u32
            COMMAND glslc src/${INPUT_FILE} -I${SHADER_INCLUDE_PATH} -DGLSL -o ${SHADER_INCLUDE_PATH}/${INPUT_FILE}.u32 -mfmt=num --target-env=vulkan1.3 -O
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/${INPUT_FILE} ${SHADER_HEADER}
    )
endfunction()

foreach (X IN LISTS SHADER_SRC)
    add_spirv_shader(${X})
endforeach ()
# ENDIF ()

set(SHADER_U32)
foreach (X IN LISTS SHADER_SRC)
    list(APPEND SHADER_U32 include/shader/${X}.u32)
endforeach ()

add_custom_target(
        shader_u32 ALL DEPENDS
        ${SHADER_U32}
)
add_library(shader INTERFACE)
add_dependencies(shader shader_u32)
target_include_directories(shader INTERFACE include)