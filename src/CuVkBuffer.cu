//
// Created by adamyuan on 4/2/24.
//

#include "CuVkBuffer.hpp"

#include <tiny-cuda-nn/config.h>

struct CuVkBuffer::CudaImpl {
	void *mapped;
	cudaExternalMemory_t ext_mem;
};

CuVkBuffer::CuVkBuffer(const myvk::Ptr<myvk::ExportBuffer> &vk_buffer) : m_vk_buffer(vk_buffer) {
	VkExternalMemoryHandleTypeFlagBits handle_type = vk_buffer->GetMemoryHandleType();
	cudaExternalMemoryHandleDesc ext_mem_handle_desc = {};

	if (handle_type == VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT)
		ext_mem_handle_desc.type = cudaExternalMemoryHandleTypeOpaqueWin32;
	else if (handle_type == VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT)
		ext_mem_handle_desc.type = cudaExternalMemoryHandleTypeOpaqueWin32Kmt;
	else if (handle_type == VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT)
		ext_mem_handle_desc.type = cudaExternalMemoryHandleTypeOpaqueFd;

	ext_mem_handle_desc.size = vk_buffer->GetSize();

#ifdef _WIN64
	ext_mem_handle_desc.handle.win32.handle = (HANDLE)vk_buffer->GetMemoryHandle();
#else
	ext_mem_handle_desc.handle.fd = (int)(uintptr_t)vk_buffer->GetMemoryHandle();
#endif
	float *mapped;
	cudaExternalMemory_t ext_mem;
	CUDA_CHECK_THROW(cudaImportExternalMemory(&ext_mem, &ext_mem_handle_desc));
	cudaExternalMemoryBufferDesc ext_mem_buffer_desc = {};
	ext_mem_buffer_desc.offset = 0;
	ext_mem_buffer_desc.size = vk_buffer->GetSize();
	ext_mem_buffer_desc.flags = 0;
	CUDA_CHECK_THROW(cudaExternalMemoryGetMappedBuffer((void **)&mapped, ext_mem, &ext_mem_buffer_desc));

	m_p_cuda_impl = new CudaImpl{
	    .mapped = mapped,
	    .ext_mem = ext_mem,
	};
}

CuVkBuffer::~CuVkBuffer() {
	cudaDestroyExternalMemory(m_p_cuda_impl->ext_mem);
	delete m_p_cuda_impl;
}

void *CuVkBuffer::GetMappedPtr() const { return m_p_cuda_impl->mapped; }
