#pragma once
#ifndef MYVK_RG_DEF_EXE_ERROR_HPP
#define MYVK_RG_DEF_EXE_ERROR_HPP

#include <myvk_rg/interface/Alias.hpp>
#include <myvk_rg/interface/Key.hpp>
#include <stdexcept>
#include <variant>

namespace myvk_rg_executor {

using namespace myvk_rg::interface;

namespace error {

struct NullResource {
	GlobalKey parent;
	inline std::string Format() const { return "Null Resource in " + parent.Format(); }
};
struct NullInput {
	GlobalKey parent;
	inline std::string Format() const { return "Null Input in " + parent.Format(); }
};
struct NullPass {
	GlobalKey parent;
	inline std::string Format() const { return "Null Pass in " + parent.Format(); }
};
struct ResourceNotFound {
	GlobalKey key;
	inline std::string Format() const { return "Resource " + key.Format() + " not found"; }
};
struct InputNotFound {
	GlobalKey key;
	inline std::string Format() const { return "Input " + key.Format() + " not found"; }
};
struct PassNotFound {
	GlobalKey key;
	inline std::string Format() const { return "Pass " + key.Format() + " not found"; }
};
struct AliasNoMatch {
	AliasBase alias;
	ResourceType actual_type;
	inline std::string Format() const {
		return "Alias source " + alias.GetSourceKey().Format() + " is not matched with type " +
		       std::to_string(static_cast<int>(actual_type));
	}
};
struct WriteToLastFrame {
	AliasBase alias;
	inline std::string Format() const { return "Write to last frame source " + alias.GetSourceKey().Format(); }
};
struct LFResourceSrcUnused {
	GlobalKey key;
	inline std::string Format() const { return "Last frame resource " + key.Format() + " has unused source resource"; }
};
struct MultipleWrite {
	AliasBase alias;
	inline std::string Format() const {
		return "Alias source " + alias.GetSourceKey().Format() + " is written multiple times";
	}
};
struct ResourceMultiInput {
	AliasBase alias;
	inline std::string Format() const {
		return "Alias source " + alias.GetSourceKey().Format() + " is input multiple times in the same pass";
	}
};
struct PassNotDAG {
	inline std::string Format() const { return "Pass cycle dependencies in Render Graph"; }
};
struct ResourceNotTree {
	inline std::string Format() const { return "Resources are not tree structured"; }
};
struct ResourceLFParent {
	GlobalKey key;
	inline std::string Format() const {
		return "Last frame resource " + key.Format() + " is referenced by another resource";
	}
};
struct ResourceExtParent {
	GlobalKey key;
	inline std::string Format() const {
		return "External resource " + key.Format() + " is referenced by another resource";
	}
};
struct ImageNotMerge {
	GlobalKey key;
	inline std::string Format() const { return "Image " + key.Format() + " failed to merge"; }
};
struct BufferNotMerge {
	GlobalKey key;
	inline std::string Format() const { return "Buffer " + key.Format() + " failed to merge"; }
};
struct DupAttachmentIndex {
	GlobalKey key;
	inline std::string Format() const { return "Duplicated attachment index with " + key.Format(); }
};
struct DupDescriptorIndex {
	GlobalKey key;
	inline std::string Format() const { return "Duplicated descriptor index with " + key.Format(); }
};
struct InvalidDescriptorArray {
	GlobalKey key;
	inline std::string Format() const { return "Invalid descriptor array in Pass " + key.Format(); }
};
struct ResourceLastInputNotRoot {
	GlobalKey key;
	inline std::string Format() const {
		return "Last inputs for resource " + key.Format() + " is not on its root resource";
	}
};

template <typename Error> struct Exception : public std::exception {
	std::string format;
	explicit Exception(Error error) : format{error.Format()} {}
	inline const char *what() const noexcept override { return format.c_str(); }
};
template <typename Error> void Throw(Error &&error) {
	throw Exception<Error>{std::forward<Error>(error)};
	// TODO: abort() if exception is disabled
}

} // namespace error

} // namespace myvk_rg_executor

#endif
