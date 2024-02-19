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
	Ptr<Buffer> m_buffer, m_scratch_buffer;
	VkAccelerationStructureKHR m_accel_struct{VK_NULL_HANDLE};

public:
	static Ptr<AccelerationStructure> Create(const Ptr<Device> &device_ptr,
	                                         const VkAccelerationStructureCreateInfoKHR &create_info);
	~AccelerationStructure() final;
	inline const Ptr<Device> &GetDevicePtr() const { return m_buffer->GetDevicePtr(); }
	inline const Ptr<Buffer> &GetBuffer() const { return m_buffer; }
};

} // namespace myvk

#endif
