"""
Prediction and XAI related utility functions.

This module encapsulates the logic for making predictions with the model
and generating the corresponding XAI (Explainable AI) visualizations.
"""

import torch
from xai_utils import (
    generate_manual_gradcam_image,
    generate_library_cam_visualization,
    generate_guided_cam,
    generate_guided_backprop_image,
    ManualGradCAM,
    LIBRARY_CAM_METHODS,
)


def get_prediction(learn, pil_image):
    """Gets the top-k predictions for a given model and image."""
    if learn is None:
        return None

    try:
        _, _, outputs_tensor = learn.predict(pil_image)
        k = min(5, len(learn.dls.vocab))
        top_k_probs, top_k_indices = torch.topk(outputs_tensor, k=k)

        predictions_list = []
        for i in range(k):
            idx_item = top_k_indices[i].item()
            prob_item = top_k_probs[i].item()
            style_name_item = learn.dls.vocab[idx_item]
            predictions_list.append(
                {
                    "style": style_name_item,
                    "confidence": prob_item,
                    "pred_idx": idx_item,
                }
            )
        return predictions_list
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None

def get_xai_visualization(
    viz_type,
    cam_method_name,
    learner,
    pil_image,
    pred_idx,
    image_identifier_for_cache,
):
    """Generates the selected XAI visualization."""
    xai_image = None
    cam_method_class = None

    if viz_type in ["Heatmap", "Guided CAM (Combinat)"]:
        full_cam_options = {
            "Grad-CAM (Manual)": ManualGradCAM,
            **LIBRARY_CAM_METHODS,
        }
        cam_method_class = full_cam_options.get(cam_method_name)

    if viz_type == "Heatmap":
        if cam_method_class == ManualGradCAM:
            xai_image = generate_manual_gradcam_image(
                _learner=learner,
                _pil_image=pil_image,
                predicted_class_idx=pred_idx,
                image_identifier_for_cache=f"{image_identifier_for_cache}_{pred_idx}",
            )
        elif cam_method_class:
            xai_image = generate_library_cam_visualization(
                _learner=learner,
                _pil_image=pil_image,
                predicted_class_idx=pred_idx,
                cam_method_class=cam_method_class,
            )

    elif viz_type == "Guided Backpropagation":
        xai_image = generate_guided_backprop_image(
            _learner=learner,
            _pil_image=pil_image,
            predicted_class_idx=pred_idx,
        )

    elif viz_type == "Guided CAM (Combinat)" and cam_method_class:
        xai_image = generate_guided_cam(
            _learner=learner,
            _pil_image=pil_image,
            predicted_class_idx=pred_idx,
            cam_method_class=cam_method_class,
        )

    return xai_image
