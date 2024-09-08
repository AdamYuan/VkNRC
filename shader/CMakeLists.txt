cmake_minimum_required(VERSION 3.15)
project(VkNRC)

find_program(GLSLC "glslc" REQUIRED)

add_library(shader INTERFACE)
target_include_directories(shader INTERFACE include)

set(SHADER_HEADER
        Constant.glsl
        RNG.glsl
        Sample.glsl
        LambertBRDF.glsl
        CookTorranceBRDF.glsl
        NRCRecord.glsl
        Scene.glsl
        NN_nv.glsl
)
list(TRANSFORM SHADER_HEADER PREPEND ${CMAKE_CURRENT_SOURCE_DIR}/src/)

function(add_shader INPUT_FILE OUTPUT_FILE OPTION)
    add_custom_command(OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/include/shader/${OUTPUT_FILE}
            COMMAND ${GLSLC} src/${INPUT_FILE} -o include/shader/${OUTPUT_FILE} -mfmt=num --target-env=vulkan1.3 ${OPTION}
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/${INPUT_FILE} ${SHADER_HEADER}
    )
    add_custom_target(shader_${OUTPUT_FILE} ALL DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/include/shader/${OUTPUT_FILE})
    add_dependencies(shader shader_${OUTPUT_FILE})
endfunction()

add_shader(vbuffer.frag vbuffer.frag.u32 "-O")
add_shader(vbuffer.vert vbuffer.vert.u32 "-O")
add_shader(path_tracer.comp path_tracer.comp.u32 "-O")
add_shader(screen.frag screen.frag.u32 "-O")
add_shader(screen.vert screen.vert.u32 "-O")
add_shader(nrc_indirect.comp nrc_indirect.comp.u32 "-O")
add_shader(nrc_inference.comp nrc_inference_16.comp.u32 "-O;-DSUBGROUP_SIZE=16")
add_shader(nrc_inference.comp nrc_inference_32.comp.u32 "-O;-DSUBGROUP_SIZE=32")
add_shader(nrc_inference.comp nrc_inference_64.comp.u32 "-O;-DSUBGROUP_SIZE=64")
add_shader(nrc_train_prepare.comp nrc_train_prepare.comp.u32 "-O")
add_shader(nrc_gradient.comp nrc_gradient_16.comp.u32 "-O;-DSUBGROUP_SIZE=16")
add_shader(nrc_gradient.comp nrc_gradient_32.comp.u32 "-O;-DSUBGROUP_SIZE=32")
add_shader(nrc_gradient.comp nrc_gradient_64.comp.u32 "-O;-DSUBGROUP_SIZE=64")
add_shader(nrc_optimize.comp nrc_optimize.comp.u32 "-O")
add_shader(nrc_optimize.comp nrc_optimize_use.comp.u32 "-O;-DWRITE_USE_WEIGHTS")
