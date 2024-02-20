//
// Created by adamyuan on 2/20/24.
//

#include "VkSceneView.hpp"

VkSceneView::VkSceneView(const myvk::Ptr<VkSceneTLAS> &scene_tlas_ptr)
    : m_scene_tlas_ptr(scene_tlas_ptr), m_scene_ptr(scene_tlas_ptr->GetScenePtr()) {
	create_descriptor();
}

void VkSceneView::create_descriptor() {

}
