#!/bin/sh
echo gradient 
glslc gradient.comp -DSUBGROUP_SIZE=32 --target-env=vulkan1.3 -O -o gradient_32.spv
echo optimize 
glslc optimize.comp -DSUBGROUP_SIZE=32 --target-env=vulkan1.3 -O -o optimize_32.spv
echo inference 
glslc inference.comp -DSUBGROUP_SIZE=32 --target-env=vulkan1.3 -O -o inference_32.spv
