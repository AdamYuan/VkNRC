# Vulkan Neural Radiance Caching
[![Windows MinGW](https://github.com/AdamYuan/VkNRC/actions/workflows/windows-mingw.yml/badge.svg)](https://github.com/AdamYuan/VkNRC/actions/workflows/windows-mingw.yml)
[![Windows MSVC](https://github.com/AdamYuan/VkNRC/actions/workflows/windows-msvc.yml/badge.svg)](https://github.com/AdamYuan/VkNRC/actions/workflows/windows-msvc.yml)

Vulkan Implementation of NVIDIA's paper [Real-time Neural Radiance Caching for Path Tracing](https://research.nvidia.com/publication/2021-06_real-time-neural-radiance-caching-path-tracing).

The Fully-Fused MLP is implemented with VK_NV_cooperative_matrix (the KHR one is too limited), and it has better backpropagation performance than the author's [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn).