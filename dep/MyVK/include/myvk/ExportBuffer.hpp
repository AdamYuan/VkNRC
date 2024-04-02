#pragma once
#ifndef MYVK_EXPORT_BUFFER_HPP
#define MYVK_EXPORT_BUFFER_HPP

#include "BufferBase.hpp"

namespace myvk {

class ExportBuffer final : public BufferBase {
private:
	Ptr<Device> m_device_ptr;
	VkDeviceMemory m_device_memory{VK_NULL_HANDLE};
	VkExternalMemoryHandleTypeFlagBits m_ext_handle_type{};
	void *m_mem_handle{nullptr};

public:
	~ExportBuffer() override;
	static const char *GetExternalMemoryExtensionName();
	static Ptr<ExportBuffer> Create(const Ptr<Device> &device, VkDeviceSize size, VkBufferUsageFlags usage,
	                                VkMemoryPropertyFlags memory_properties,
	                                const std::vector<Ptr<Queue>> &access_queues = {});
	inline void *GetMemoryHandle() const { return m_mem_handle; }
	inline VkDeviceMemory GetDeviceMemoryHandle() const { return m_device_memory; }
	inline VkExternalMemoryHandleTypeFlagBits GetMemoryHandleType() const { return m_ext_handle_type; }

	inline const Ptr<Device> &GetDevicePtr() const override { return m_device_ptr; }
};

} // namespace myvk

#endif
