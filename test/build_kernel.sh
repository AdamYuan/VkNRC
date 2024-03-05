#!/bin/sh
echo evaluate_NV
glslc evaluate_NV.comp -DSUBGROUP_SIZE=32 --target-env=vulkan1.3 -O -o evaluate_32.spv
# glslc evaluate_old.comp -DSUBGROUP_SIZE=32 --target-env=vulkan1.3 -O -o evaluate_32.spv
# glslc evaluate_NV.comp -DSUBGROUP_SIZE=64 --target-env=vulkan1.3 -O -o evaluate_64.spv
echo train_NV
glslc train_NV.comp -DSUBGROUP_SIZE=32 --target-env=vulkan1.3 -O -o train_32.spv
# glslc train_NV.comp -DSUBGROUP_SIZE=64 --target-env=vulkan1.3 -O -o train_64.spv
glslc empty.comp --target-env=vulkan1.3 -O -o empty.spv
