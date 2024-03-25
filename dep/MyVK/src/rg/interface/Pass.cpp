#include <myvk_rg/interface/Pass.hpp>

#include <myvk_rg/interface/RenderGraph.hpp>

namespace myvk_rg::interface {

uint32_t GraphicsPassBase::GetSubpass() const { return executor::Executor::GetSubpass(this); }
const myvk::Ptr<myvk::RenderPass> &GraphicsPassBase::GetVkRenderPass() const {
	return GetRenderGraphPtr()->GetExecutor()->GetVkRenderPass(this);
}
const myvk::Ptr<myvk::DescriptorSetLayout> &GraphicsPassBase::GetVkDescriptorSetLayout() const {
	return executor::Executor::GetVkDescriptorSetLayout(this);
}
const myvk::Ptr<myvk::DescriptorSet> &GraphicsPassBase::GetVkDescriptorSet() const {
	return executor::Executor::GetVkDescriptorSet(this);
}
const myvk::Ptr<myvk::DescriptorSetLayout> &ComputePassBase::GetVkDescriptorSetLayout() const {
	return executor::Executor::GetVkDescriptorSetLayout(this);
}
const myvk::Ptr<myvk::DescriptorSet> &ComputePassBase::GetVkDescriptorSet() const {
	return executor::Executor::GetVkDescriptorSet(this);
}

const ImageBase *GraphicsPassBase::GetInputImage(const PoolKey &input_key) const {
	return executor::Executor::GetInputImage(GetInput(input_key));
}
const BufferBase *GraphicsPassBase::GetInputBuffer(const PoolKey &input_key) const {
	return executor::Executor::GetInputBuffer(GetInput(input_key));
}
const ImageBase *ComputePassBase::GetInputImage(const PoolKey &input_key) const {
	return executor::Executor::GetInputImage(GetInput(input_key));
}
const BufferBase *ComputePassBase::GetInputBuffer(const PoolKey &input_key) const {
	return executor::Executor::GetInputBuffer(GetInput(input_key));
}
const ImageBase *TransferPassBase::GetInputImage(const PoolKey &input_key) const {
	return executor::Executor::GetInputImage(GetInput(input_key));
}
const BufferBase *TransferPassBase::GetInputBuffer(const PoolKey &input_key) const {
	return executor::Executor::GetInputBuffer(GetInput(input_key));
}

const myvk::Ptr<myvk::PipelineBase> &GraphicsPassBase::GetVkPipeline() const {
	return executor::Executor::GetVkPipeline(this);
}
const myvk::Ptr<myvk::PipelineBase> &ComputePassBase::GetVkPipeline() const {
	return executor::Executor::GetVkPipeline(this);
}

} // namespace myvk_rg::interface
