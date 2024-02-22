//
// Created by adamyuan on 2/18/24.
//
#include <myvk/AccelerationStructure.hpp>

namespace myvk {

Ptr<AccelerationStructure> AccelerationStructure::Create(const Ptr<BufferBase> &buffer,
                                                         VkAccelerationStructureCreateInfoKHR create_info) {
	create_info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
	create_info.buffer = buffer->GetHandle();

	VkAccelerationStructureKHR accel_struct = VK_NULL_HANDLE;
	if (vkCreateAccelerationStructureKHR(buffer->GetDevicePtr()->GetHandle(), &create_info, nullptr, &accel_struct) !=
	    VK_SUCCESS)
		return nullptr;

	VkAccelerationStructureDeviceAddressInfoKHR device_address_info = {
	    .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR,
	    .pNext = nullptr,
	    .accelerationStructure = accel_struct};

	VkDeviceAddress device_address =
	    vkGetAccelerationStructureDeviceAddressKHR(buffer->GetDevicePtr()->GetHandle(), &device_address_info);

	auto ret = MakePtr<AccelerationStructure>();
	ret->m_buffer = buffer;
	ret->m_accel_struct = accel_struct;
	ret->m_device_address = device_address;
	ret->m_offset = create_info.offset;
	ret->m_size = create_info.size;
	return ret;
}

Ptr<AccelerationStructure> AccelerationStructure::Create(const Ptr<Device> &device, VkDeviceSize size,
                                                         VkAccelerationStructureTypeKHR type,
                                                         const std::vector<Ptr<Queue>> &access_queues) {
	return Create(myvk::Buffer::Create(device, size, 0, VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR,
	                                   VMA_MEMORY_USAGE_AUTO, access_queues),
	              {.size = size, .type = type});
}

AccelerationStructure::~AccelerationStructure() {
	if (m_accel_struct != VK_NULL_HANDLE)
		vkDestroyAccelerationStructureKHR(m_buffer->GetDevicePtr()->GetHandle(), m_accel_struct, nullptr);
}

} // namespace myvk
