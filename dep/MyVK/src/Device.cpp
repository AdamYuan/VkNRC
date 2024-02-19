#include "myvk/Device.hpp"
#include "myvk/Queue.hpp"
#include <algorithm>
#include <myvk/Allocator.hpp>
#include <ranges>
#include <set>

namespace myvk {

class QueueSelectionResolver {
private:
	std::vector<QueueSelection> m_queue_selections;
	std::map<uint32_t, std::vector<std::vector<QueueSelection *>>> m_queue_creations;

public:
	QueueSelectionResolver(const Ptr<PhysicalDevice> &physical_device, std::vector<QueueSelection> &&queue_selections);

	void EnumerateDeviceQueueCreateInfos(std::vector<VkDeviceQueueCreateInfo> *out_create_infos,
	                                     std::vector<float> *out_priorities) const;
	void FetchDeviceQueues(const Ptr<Device> &device) const;
};

Device::~Device() {
	if (m_pipeline_cache)
		vkDestroyPipelineCache(m_device, m_pipeline_cache, nullptr);
	if (m_allocator)
		vmaDestroyAllocator(m_allocator);
	if (m_dev_addr_allocator)
		vmaDestroyAllocator(m_dev_addr_allocator);
	if (m_device)
		vkDestroyDevice(m_device, nullptr);
}

Ptr<Device> Device::Create(const Ptr<PhysicalDevice> &physical_device, const QueueSelectorFunc &queue_selector_func,
                           const PhysicalDeviceFeatures &features, const std::vector<const char *> &extensions) {
	auto ret = std::make_shared<Device>();
	ret->m_physical_device_ptr = physical_device;

	std::vector<QueueSelection> queue_selections = queue_selector_func(physical_device);
	if (queue_selections.empty())
		return nullptr;
	QueueSelectionResolver queue_resolver{physical_device, std::move(queue_selections)};
	std::vector<VkDeviceQueueCreateInfo> queue_create_infos;
	std::vector<float> queue_priorities;
	queue_resolver.EnumerateDeviceQueueCreateInfos(&queue_create_infos, &queue_priorities);

	if (ret->create_device(queue_create_infos, extensions, features) != VK_SUCCESS)
		return nullptr;
	volkLoadDevice(ret->m_device);

	ret->m_allocator = Allocator::CreateHandle(ret, {});
	if (ret->m_allocator == VK_NULL_HANDLE)
		return nullptr;
	ret->m_dev_addr_allocator = Allocator::CreateHandle(ret, {.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT});
	if (ret->m_dev_addr_allocator == VK_NULL_HANDLE)
		return nullptr;
	if (ret->create_pipeline_cache() != VK_SUCCESS)
		return nullptr;
	ret->m_features = features;
	queue_resolver.FetchDeviceQueues(ret);
	return ret;
}

VkResult Device::create_device(const std::vector<VkDeviceQueueCreateInfo> &queue_create_infos,
                               const std::vector<const char *> &extensions, const PhysicalDeviceFeatures &features) {
	VkDeviceCreateInfo create_info = {};
	create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
	create_info.pQueueCreateInfos = queue_create_infos.data();
	create_info.queueCreateInfoCount = queue_create_infos.size();
	// features.robustBufferAccess = VK_FALSE; // from ARM/AMD best practice
	create_info.pEnabledFeatures = &features.vk10;
	create_info.enabledExtensionCount = extensions.size();
	create_info.ppEnabledExtensionNames = extensions.data();
	create_info.enabledLayerCount = 0;
	create_info.pNext = &features.vk11;

	return vkCreateDevice(m_physical_device_ptr->GetHandle(), &create_info, nullptr, &m_device);
}

VkResult Device::create_pipeline_cache() {
	VkPipelineCacheCreateInfo create_info = {};
	create_info.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;

	return vkCreatePipelineCache(m_device, &create_info, nullptr, &m_pipeline_cache);
}

VkResult Device::WaitIdle() const { return vkDeviceWaitIdle(m_device); }

QueueSelectionResolver::QueueSelectionResolver(const Ptr<PhysicalDevice> &physical_device,
                                               std::vector<QueueSelection> &&queue_selections)
    : m_queue_selections{std::move(queue_selections)} {
	// transform index_specifiers to queue_indices
	std::map<uint32_t, std::set<uint32_t>> queue_sets;
	std::map<uint32_t, std::map<uint32_t, uint32_t>> queue_tables;

	for (const auto &i : m_queue_selections)
		queue_sets[i.GetFamily()].insert(i.GetIndexSpecifier());

	for (const auto &i : queue_sets) {
		uint32_t queue_count = physical_device->GetQueueFamilyProperties()[i.first].queueCount;
		m_queue_creations[i.first].resize(std::min((uint32_t)i.second.size(), queue_count));

		std::map<uint32_t, uint32_t> table;
		uint32_t cnt = 0;
		for (uint32_t x : i.second)
			table[x] = (cnt++) % queue_count;
		queue_tables[i.first] = std::move(table);
	}

	for (auto &i : m_queue_selections) {
		uint32_t queue_index = queue_tables[i.GetFamily()][i.GetIndexSpecifier()];
		m_queue_creations[i.GetFamily()][queue_index].push_back(&i);
	}
}
void QueueSelectionResolver::EnumerateDeviceQueueCreateInfos(std::vector<VkDeviceQueueCreateInfo> *out_create_infos,
                                                             std::vector<float> *out_priorities) const {
	out_create_infos->clear();
	out_priorities->clear();

	if (m_queue_creations.empty())
		return;

	uint32_t max_queue_count = 0;
	for (const auto &i : m_queue_creations) {
		if (i.second.size() > max_queue_count)
			max_queue_count = i.second.size();
	}
	out_priorities->resize(max_queue_count, 1.0f);

	for (const auto &i : m_queue_creations) {
		VkDeviceQueueCreateInfo info = {};
		info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
		info.queueFamilyIndex = i.first;
		info.queueCount = i.second.size();
		info.pQueuePriorities = out_priorities->data();
		out_create_infos->push_back(info);
	}
}
void QueueSelectionResolver::FetchDeviceQueues(const Ptr<Device> &device) const {
	for (const auto &creation : m_queue_creations) {
		uint32_t family = creation.first;
		for (uint32_t index = 0; index < creation.second.size(); ++index) {
			Ptr<UniqueQueue> unique_queue = UniqueQueue::Create(device, family, index);
			for (QueueSelection *selection : creation.second[index]) {
				if (selection->IsPresentQueue())
					selection->GetPresentQueueTargetRef() =
					    PresentQueue::Create(unique_queue, selection->GetSurfacePtr());
				else {
					selection->GetQueueTargetRef() = Queue::Create(unique_queue);
				}
			}
		}
	}
}

} // namespace myvk
