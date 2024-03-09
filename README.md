# Vulkan Neural Radiance Caching
[![Windows MinGW](https://github.com/AdamYuan/VkNRC/actions/workflows/windows-mingw.yml/badge.svg)](https://github.com/AdamYuan/VkNRC/actions/workflows/windows-mingw.yml)

Vulkan Implementation of NVIDIA's paper Real-time Neural Radiance Caching for Path Tracing.

The Fully-Fused MLP is implemented with VK_NV_cooperative_matrix (the KHR one is too limited), and it has better backpropagation performance than the author's [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn).