#include <myvk/Allocator.hpp>

namespace myvk {

VmaAllocator Allocator::CreateHandle(const Ptr<Device> &device_ptr, const VmaAllocatorCreateInfo &create_info) {
	VmaVulkanFunctions vk_funcs = {
	    /// Required when using VMA_DYNAMIC_VULKAN_FUNCTIONS.
	    vkGetInstanceProcAddr,
	    /// Required when using VMA_DYNAMIC_VULKAN_FUNCTIONS.
	    vkGetDeviceProcAddr,
	    vkGetPhysicalDeviceProperties,
	    vkGetPhysicalDeviceMemoryProperties,
	    vkAllocateMemory,
	    vkFreeMemory,
	    vkMapMemory,
	    vkUnmapMemory,
	    vkFlushMappedMemoryRanges,
	    vkInvalidateMappedMemoryRanges,
	    vkBindBufferMemory,
	    vkBindImageMemory,
	    vkGetBufferMemoryRequirements,
	    vkGetImageMemoryRequirements,
	    vkCreateBuffer,
	    vkDestroyBuffer,
	    vkCreateImage,
	    vkDestroyImage,
	    vkCmdCopyBuffer,
#if VMA_DEDICATED_ALLOCATION || VMA_VULKAN_VERSION >= 1001000
	    /// Fetch "vkGetBufferMemoryRequirements2" on Vulkan >= 1.1, fetch "vkGetBufferMemoryRequirements2KHR" when
	    /// using
	    /// VK_KHR_dedicated_allocation extension.
	    vkGetBufferMemoryRequirements2KHR,
	    /// Fetch "vkGetImageMemoryRequirements 2" on Vulkan >= 1.1, fetch "vkGetImageMemoryRequirements2KHR" when using
	    /// VK_KHR_dedicated_allocation extension.
	    vkGetImageMemoryRequirements2KHR,
#endif
#if VMA_BIND_MEMORY2 || VMA_VULKAN_VERSION >= 1001000
	    /// Fetch "vkBindBufferMemory2" on Vulkan >= 1.1, fetch "vkBindBufferMemory2KHR" when using VK_KHR_bind_memory2
	    /// extension.
	    vkBindBufferMemory2KHR,
	    /// Fetch "vkBindImageMemory2" on Vulkan >= 1.1, fetch "vkBindImageMemory2KHR" when using VK_KHR_bind_memory2
	    /// extension.
	    vkBindImageMemory2KHR,
#endif
#if VMA_MEMORY_BUDGET || VMA_VULKAN_VERSION >= 1001000
	    vkGetPhysicalDeviceMemoryProperties2KHR,
#endif
#if VMA_VULKAN_VERSION >= 1003000
	    /// Fetch from "vkGetDeviceBufferMemoryRequirements" on Vulkan >= 1.3, but you can also fetch it from
	    /// "vkGetDeviceBufferMemoryRequirementsKHR" if you enabled extension VK_KHR_maintenance4.
	    vkGetDeviceBufferMemoryRequirements,
	    /// Fetch from "vkGetDeviceImageMemoryRequirements" on Vulkan >= 1.3, but you can also fetch it from
	    /// "vkGetDeviceImageMemoryRequirementsKHR" if you enabled extension VK_KHR_maintenance4.
	    vkGetDeviceImageMemoryRequirements,
#endif
	};

	VmaAllocatorCreateInfo new_create_info = create_info;
	new_create_info.instance = device_ptr->GetPhysicalDevicePtr()->GetInstancePtr()->GetHandle();
	new_create_info.device = device_ptr->GetHandle();
	new_create_info.physicalDevice = device_ptr->GetPhysicalDevicePtr()->GetHandle();
	new_create_info.pVulkanFunctions = &vk_funcs;
	new_create_info.vulkanApiVersion = VK_API_VERSION_1_3;

	VmaAllocator allocator = VK_NULL_HANDLE;
	if (vmaCreateAllocator(&new_create_info, &allocator) != VK_SUCCESS)
		return VK_NULL_HANDLE;
	return allocator;
}
Ptr<Allocator> Allocator::Create(const Ptr<Device> &device_ptr, const VmaAllocatorCreateInfo &create_info) {
	VmaAllocator allocator = CreateHandle(device_ptr, create_info);
	if (allocator == VK_NULL_HANDLE)
		return nullptr;
	auto ret = myvk::MakePtr<Allocator>();
	ret->m_allocator = allocator;
	ret->m_device_ptr = device_ptr;
	return ret;
}

Allocator::~Allocator() {
	if (m_allocator != VK_NULL_HANDLE)
		vmaDestroyAllocator(m_allocator);
}

} // namespace myvk