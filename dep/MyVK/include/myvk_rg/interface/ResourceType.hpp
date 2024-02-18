#pragma once
#ifndef MYVK_RG_RESOURCE_TYPE_HPP
#define MYVK_RG_RESOURCE_TYPE_HPP

#include <cinttypes>

namespace myvk_rg::interface {
// Resource Base and Types
enum class ResourceType : uint8_t { kImage, kBuffer };
} // namespace myvk_rg::interface

#endif // MYVK_RESOURCETYPE_HPP
