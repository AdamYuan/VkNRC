#include <myvk_rg/interface/Resource.hpp>

#include <myvk_rg/interface/RenderGraph.hpp>

namespace myvk_rg::interface {

const myvk::Ptr<myvk::ImageView> &LastFrameImage::GetVkImageView() const {
	return GetRenderGraphPtr()->GetExecutor()->GetVkImageView(this);
}
const myvk::Ptr<myvk::ImageView> &CombinedImage::GetVkImageView() const {
	return GetRenderGraphPtr()->GetExecutor()->GetVkImageView(this);
}
const myvk::Ptr<myvk::ImageView> &ManagedImage::GetVkImageView() const {
	return GetRenderGraphPtr()->GetExecutor()->GetVkImageView(this);
}

const myvk::Ptr<myvk::BufferBase> &LastFrameBuffer::GetVkBuffer() const {
	return GetRenderGraphPtr()->GetExecutor()->GetVkBuffer(this);
}
const myvk::Ptr<myvk::BufferBase> &ManagedBuffer::GetVkBuffer() const {
	return GetRenderGraphPtr()->GetExecutor()->GetVkBuffer(this);
}

void *ManagedBuffer::GetMappedData() const {
	return GetRenderGraphPtr()->GetExecutor()->GetMappedData(this);
}

} // namespace myvk_rg::interface
