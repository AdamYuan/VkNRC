#include <myvk_rg/interface/Resource.hpp>

#include <myvk_rg/interface/RenderGraph.hpp>

namespace myvk_rg::interface {

const myvk::Ptr<myvk::ImageView> &CombinedImage::GetVkImageView() const {
	return executor::Executor::GetVkImageView(this);
}
const myvk::Ptr<myvk::ImageView> &ManagedImage::GetVkImageView() const {
	return executor::Executor::GetVkImageView(this);
}

const BufferView &ManagedBuffer::GetBufferView() const { return executor::Executor::GetBufferView(this); }
const BufferView &CombinedBuffer::GetBufferView() const { return executor::Executor::GetBufferView(this); }
void *CombinedBuffer::GetMappedData() const { return executor::Executor::GetMappedData(this); }
void *ManagedBuffer::GetMappedData() const { return executor::Executor::GetMappedData(this); }

} // namespace myvk_rg::interface
