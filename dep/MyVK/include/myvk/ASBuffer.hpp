//
// Created by adamyuan on 2/18/24.
//

#pragma once
#ifndef MYVK_ACCSTRUCTBUFFER_HPP
#define MYVK_ACCSTRUCTBUFFER_HPP

#include "BufferBase.hpp"
#include "Queue.hpp"

namespace myvk {

// Buffer for Acceleration Structure related things
class ASBuffer final : public BufferBase {
private:
	Ptr<Device> m_device_ptr;

	VmaAllocation m_allocation{VK_NULL_HANDLE};
	void *m_mapped_ptr{};
	VkDeviceAddress m_address{};

public:
	static Ptr<ASBuffer> Create(const Ptr<Device> &device, const VkBufferCreateInfo &create_info,
	                            VmaAllocationCreateFlags allocation_flags,
	                            VmaMemoryUsage memory_usage = VMA_MEMORY_USAGE_AUTO,
	                            const std::vector<Ptr<Queue>> &access_queues = {});

	static Ptr<ASBuffer> Create(const Ptr<Device> &device, VkDeviceSize size, VmaAllocationCreateFlags allocation_flags,
	                            VkBufferUsageFlags buffer_usage, VmaMemoryUsage memory_usage = VMA_MEMORY_USAGE_AUTO,
	                            const std::vector<Ptr<Queue>> &access_queues = {});

	inline void *GetMappedData() const { return m_mapped_ptr; }
	inline VkDeviceAddress GetAddress() const { return m_address; }

	inline const Ptr<Device> &GetDevicePtr() const final { return m_device_ptr; }

	~ASBuffer() final;
};

} // namespace myvk

#endif
