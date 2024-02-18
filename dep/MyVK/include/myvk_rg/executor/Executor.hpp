#pragma once
#ifndef MYVK_RG_DEFAULT_EXECUTOR_HPP
#define MYVK_RG_DEFAULT_EXECUTOR_HPP

#include <myvk/CommandBuffer.hpp>
#include <myvk_rg/interface/Object.hpp>
#include <myvk_rg/interface/Pass.hpp>
#include <myvk_rg/interface/Resource.hpp>

namespace myvk_rg::executor {

class Executor final : public interface::ObjectBase {
private:
	struct CompileInfo;
	uint8_t m_compile_flags{};
	CompileInfo *m_p_compile_info;
	bool m_flip{}, m_lf_init{};

	void compile(const interface::RenderGraphBase *p_render_graph, const myvk::Ptr<myvk::Queue> &queue);

public:
	explicit Executor(interface::Parent parent);
	~Executor() final;

	void OnEvent(interface::ObjectBase *p_object, interface::Event event);
	void CmdExecute(const interface::RenderGraphBase *p_render_graph,
	                const myvk::Ptr<myvk::CommandBuffer> &command_buffer);

	const myvk::Ptr<myvk::ImageView> &GetVkImageView(const interface::ManagedImage *p_managed_image) const;
	const myvk::Ptr<myvk::ImageView> &GetVkImageView(const interface::LastFrameImage *p_lf_image) const;
	const myvk::Ptr<myvk::ImageView> &GetVkImageView(const interface::CombinedImage *p_combined_image) const;
	const myvk::Ptr<myvk::BufferBase> &GetVkBuffer(const interface::ManagedBuffer *p_managed_buffer) const;
	const myvk::Ptr<myvk::BufferBase> &GetVkBuffer(const interface::LastFrameBuffer *p_lf_buffer) const;
	void *GetMappedData(const interface::ManagedBuffer *p_managed_buffer) const;
	void *GetMappedData(const interface::LastFrameBuffer *p_lf_buffer) const;
	static uint32_t GetSubpass(const interface::PassBase *p_pass);
	const myvk::Ptr<myvk::RenderPass> &GetVkRenderPass(const interface::PassBase *p_pass) const;
	static const myvk::Ptr<myvk::DescriptorSetLayout> &GetVkDescriptorSetLayout(const interface::PassBase *p_pass) ;
	const myvk::Ptr<myvk::DescriptorSet> &GetVkDescriptorSet(const interface::PassBase *p_pass) const;
	static const interface::ImageBase *GetInputImage(const interface::InputBase *p_input);
	static const interface::BufferBase *GetInputBuffer(const interface::InputBase *p_input);
};

} // namespace myvk_rg::executor

#endif
