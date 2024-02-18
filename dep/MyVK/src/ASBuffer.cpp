//
// Created by adamyuan on 2/18/24.
//
#include <myvk/ASBuffer.hpp>

#include <set>

namespace myvk {

Ptr<ASBuffer> ASBuffer::Create(const Ptr<Device> &device, VkDeviceSize size, VmaAllocationCreateFlags allocation_flags,
                               VkBufferUsageFlags buffer_usage, VmaMemoryUsage memory_usage,
                               const std::vector<Ptr<Queue>> &access_queues) {
	VkBufferCreateInfo create_info = {};
	create_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	create_info.size = size;
	create_info.usage = buffer_usage;

	return Create(device, create_info, allocation_flags, memory_usage, access_queues);
}

Ptr<ASBuffer> ASBuffer::Create(const Ptr<Device> &device, const VkBufferCreateInfo &create_info,
                               VmaAllocationCreateFlags allocation_flags, VmaMemoryUsage memory_usage,
                               const std::vector<Ptr<Queue>> &access_queues) {
	auto ret = std::make_shared<ASBuffer>();
	ret->m_device_ptr = device;
	ret->m_size = create_info.size;

	std::set<uint32_t> queue_family_set;
	for (auto &i : access_queues)
		queue_family_set.insert(i->GetFamilyIndex());
	std::vector<uint32_t> queue_families{queue_family_set.begin(), queue_family_set.end()};

	VkBufferCreateInfo new_info{create_info};
	if (queue_families.size() <= 1) {
		new_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	} else {
		new_info.sharingMode = VK_SHARING_MODE_CONCURRENT;
		new_info.queueFamilyIndexCount = queue_families.size();
		new_info.pQueueFamilyIndices = queue_families.data();
	}
	// Add Required Usage for AS
	new_info.usage |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;

	VmaAllocationCreateInfo alloc_create_info = {};
	alloc_create_info.usage = memory_usage;
	alloc_create_info.flags = allocation_flags;

	VmaAllocationInfo allocation_info;

	if (vmaCreateBuffer(device->GetASAllocatorHandle(), &new_info, &alloc_create_info, &ret->m_buffer,
	                    &ret->m_allocation, &allocation_info) != VK_SUCCESS)
		return nullptr;
	ret->m_mapped_ptr = allocation_info.pMappedData;

	VkBufferDeviceAddressInfo device_address_info = {.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
	                                                 .buffer = ret->m_buffer};
	ret->m_address = vkGetBufferDeviceAddress(device->GetHandle(), &device_address_info);

	return ret;
}

ASBuffer::~ASBuffer() {
	if (m_buffer != VK_NULL_HANDLE)
		vmaDestroyBuffer(m_device_ptr->GetASAllocatorHandle(), m_buffer, m_allocation);
}

} // namespace myvk
