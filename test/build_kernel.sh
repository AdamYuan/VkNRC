#!/bin/sh
glslc evaluate.comp -DSUBGROUP_SIZE=32 --target-env=vulkan1.3 -O -o evaluate_32.spv
glslc evaluate.comp -DSUBGROUP_SIZE=64 --target-env=vulkan1.3 -O -o evaluate_64.spv
