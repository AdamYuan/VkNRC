//
// Created by adamyuan on 4/2/24.
//

#pragma once
#ifndef VKNRC_CUVKBUFFER_HPP
#define VKNRC_CUVKBUFFER_HPP

#include <myvk/ExportBuffer.hpp>

class CuVkBuffer {
private:
	myvk::Ptr<myvk::ExportBuffer> m_vk_buffer;
	struct CudaImpl;
	CudaImpl *m_p_cuda_impl;

public:
	explicit CuVkBuffer(const myvk::Ptr<myvk::ExportBuffer> &vk_buffer);
	inline const auto &GetVkBuffer() const { return m_vk_buffer; }
	void *GetMappedPtr() const;
	template <typename T> inline T *GetMappedPtr() const { return static_cast<T *>(GetMappedPtr()); }
	~CuVkBuffer();
};

#endif // VKNRC_CUVKBUFFER_HPP
