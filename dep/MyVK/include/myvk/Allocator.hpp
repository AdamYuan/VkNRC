#pragma once
#ifndef MYVK_ALLOCATOR_HPP
#define MYVK_ALLOCATOR_HPP

#include "DeviceObjectBase.hpp"
#include <vk_mem_alloc.h>

namespace myvk {

class Allocator : public DeviceObjectBase {
protected:
	Ptr<Device> m_device_ptr;
	VmaAllocator m_allocator{VK_NULL_HANDLE};

public:
	static VmaAllocator CreateHandle(const Ptr<Device> &device_ptr, const VmaAllocatorCreateInfo &create_info);
	static Ptr<Allocator> Create(const Ptr<Device> &device_ptr, const VmaAllocatorCreateInfo &create_info);
	inline static Ptr<Allocator> Create(const Ptr<Device> &device_ptr, VmaAllocatorCreateFlags flags) {
		return Create(device_ptr, VmaAllocatorCreateInfo{.flags = flags});
	}
	inline static Ptr<Allocator> CreateForAS(const Ptr<Device> &device_ptr) {
		return Create(device_ptr, VmaAllocatorCreateInfo{.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT});
	}
	const Ptr<Device> &GetDevicePtr() const final { return m_device_ptr; }
	VmaAllocator GetHandle() const { return m_allocator; }
	~Allocator() override;
};

} // namespace myvk

#endif
