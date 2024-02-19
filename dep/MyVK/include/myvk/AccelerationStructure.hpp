//
// Created by adamyuan on 2/18/24.
//

#pragma once
#ifndef MYVK_ACCELERATIONSTRUCTURE_HPP
#define MYVK_ACCELERATIONSTRUCTURE_HPP

#include "Buffer.hpp"

namespace myvk {

class AccelerationStructure final : public DeviceObjectBase {
private:
	Ptr<BufferBase> m_buffer;
	VkAccelerationStructureKHR m_accel_struct{VK_NULL_HANDLE};
	VkDeviceAddress m_device_address{};

public:
	static Ptr<AccelerationStructure> Create(const Ptr<BufferBase> &buffer,
	                                         VkAccelerationStructureCreateInfoKHR create_info);
	static Ptr<AccelerationStructure> Create(const Ptr<Device> &device, VkDeviceSize size,
	                                         VkAccelerationStructureTypeKHR type,
	                                         VkAccelerationStructureCreateFlagsKHR create_flags = 0);
	~AccelerationStructure() final;
	inline const Ptr<Device> &GetDevicePtr() const { return m_buffer->GetDevicePtr(); }
	inline const Ptr<BufferBase> &GetBuffer() const { return m_buffer; }
	inline VkDeviceAddress GetDeviceAddress() const { return m_device_address; }
	inline VkAccelerationStructureKHR GetHandle() const { return m_accel_struct; }
};

} // namespace myvk

#endif
