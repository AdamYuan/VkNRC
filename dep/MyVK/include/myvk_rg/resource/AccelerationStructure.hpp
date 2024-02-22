//
// Created by adamyuan on 2/21/24.
//

#pragma once
#ifndef MYVK_RG_RESOURCE_ACCELERATION_STRUCTURE_HPP
#define MYVK_RG_RESOURCE_ACCELERATION_STRUCTURE_HPP

#include <myvk/AccelerationStructure.hpp>
#include <myvk_rg/RenderGraph.hpp>

namespace myvk_rg {
class AccelerationStructure final : public myvk_rg::ExternalBufferBase {
private:
	BufferView m_buffer_view;

public:
	inline AccelerationStructure(myvk_rg::Parent parent, const myvk::Ptr<myvk::AccelerationStructure> &as)
	    : myvk_rg::ExternalBufferBase(parent) {
		SetSyncType(ExternalSyncType::kLastFrame);
		SetAS(as);
	}
	inline AccelerationStructure(myvk_rg::Parent parent) : myvk_rg::ExternalBufferBase(parent) {
		SetSyncType(ExternalSyncType::kLastFrame);
	}
	inline ~AccelerationStructure() final = default;
	inline void SetAS(const myvk::Ptr<myvk::AccelerationStructure> &as) {
		m_buffer_view = {.buffer = as->GetBuffer(), .offset = as->GetOffset(), .size = as->GetSize(), .data = as};
	}
	inline const BufferView &GetBufferView() const final { return m_buffer_view; }
};

} // namespace myvk_rg

#endif // MYVK_ACCELERATIONSTRUCTURE_HPP
