import streamlit as st
from PIL import Image
import cv2
import numpy as np
import torch
import torch.nn.functional as F

from fastai.vision.core import TensorImage, PILImage
import fastai.vision.core as fv_core 
from fastai.vision.data import imagenet_stats, Normalize
import torchvision.transforms.functional as TF 

from pytorch_grad_cam import (
    GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM,
    FullGrad, ShapleyCAM, FEM, KPCA_CAM, HiResCAM, GradCAMElementWise, 
    EigenGradCAM, LayerCAM,
    GuidedBackpropReLUModel 
)
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image, deprocess_image

TEXT_RO_XAI = {
    "xai_placeholder": "Vizualizarea nu a putut fi generată.",
    "xai_generating": "Se generează explicația vizuală ...",
}

def get_edge_map(pil_image, threshold1=50, threshold2=150):
    """
    Convertește o imagine PIL într-o schiță a contururilor folosind algoritmul Canny.
    """
    # Convertește imaginea PIL în formatul OpenCV (numpy array, BGR)
    img_np = np.array(pil_image)
    
    # Convertește la tonuri de gri
    gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
    
    # Aplică un filtru de netezire pentru a reduce zgomotul
    gray_blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Detectează contururile folosind algoritmul Canny
    edges = cv2.Canny(gray_blurred, threshold1, threshold2)
    
    # Invertează culorile (contururi albe pe fundal negru) și convertește înapoi la 3 canale
    edge_map_3_channels = np.stack([edges]*3, axis=-1)
    
    return edge_map_3_channels

class ManualGradCAM:
    def __init__(self, model, layer):
        self.model = model
        self.layer = layer
        self.features = None
        self.gradients = None
        self.hook_a = None
        self.hook_g = None
        self._register_hooks()

    def _hook_a_fn(self, module, input, output):
        self.features = output.detach()

    def _hook_g_fn(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def _register_hooks(self):
        self.hook_a = self.layer.register_forward_hook(self._hook_a_fn)
        self.hook_g = self.layer.register_full_backward_hook(self._hook_g_fn)

    def __call__(self, x, index=None):
        self.model.eval()
        device = next(self.model.parameters()).device
        x = x.to(device)

        output = self.model(x)
        if index is None:
            index = torch.argmax(output)

        self.model.zero_grad()
        output[0, index].backward(retain_graph=True)

        if self.gradients is None or self.features is None:
            return None

        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.features, dim=1, keepdim=True)
        cam = F.relu(cam)

        cam_min = torch.min(cam)
        cam_max = torch.max(cam)
        if cam_max == cam_min or (cam_max - cam_min < 1e-8):
            return np.zeros((x.shape[2], x.shape[3]), dtype=np.float32)

        cam_normalized = (cam - cam_min) / (cam_max - cam_min)
        return cam_normalized.squeeze().cpu().numpy()

    def remove_hooks(self):
        if self.hook_a: self.hook_a.remove()
        if self.hook_g: self.hook_g.remove()
        self.hook_a = None
        self.hook_g = None

def model_device(model):
    return next(model.parameters()).device

LIBRARY_CAM_METHODS = {
    # "GradCAM": GradCAM, deja făcut manual
    "GradCAM++": GradCAMPlusPlus,
    # "ScoreCAM": ScoreCAM, îi ia prea mult
    # "XGradCAM": XGradCAM, nici ăsta nu dă ce trebe
    # "AblationCAM": AblationCAM, îi ia prea mult
    "EigenCAM": EigenCAM,
    # "EigenGradCAM": EigenGradCAM, nu dă ce trebe
    "LayerCAM": LayerCAM,
    # "FullGrad": FullGrad, îi ia prea mult
    # "KPCA_CAM": KPCA_CAM, nu dă ce trebuie
    "FEM": FEM,
    "ShapleyCAM": ShapleyCAM,
    # "HiResCAM": HiResCAM, nu dă ce trebuie
    # "GradCAMElementWise": GradCAMElementWise
}

def preprocess_image(_learner, _pil_image):
    """Procesează o imagine PIL folosind pipeline-ul learner-ului."""
    img_pil_rgb = _pil_image.convert("RGB")
    test_dl = _learner.dls.test_dl([img_pil_rgb], with_labels=False, device=_learner.dls.device)
    input_tensor_batch, = test_dl.one_batch()
    return input_tensor_batch, img_pil_rgb

def find_target_layer(_learner):
    """Găsește automat un strat țintă potrivit pentru CAM."""
    try:
        if hasattr(_learner.model[0][7][-1], 'conv3'):
             return _learner.model[0][7][-1] # Returnează blocul, nu conv3, pentru compatibilitate mai largă
    except: pass
    for m in reversed(list(_learner.model.modules())):
        if isinstance(m, torch.nn.Conv2d):
            return m
    return None

@st.cache_data
def generate_manual_gradcam_image(_pil_image, _learner, predicted_class_idx, image_identifier_for_cache):
    if _learner is None:
        st.warning(TEXT_RO_XAI["xai_placeholder"] + " (Model invalid)")
        return _pil_image

    try:
        img_pil_rgb = _pil_image.convert("RGB")
        
        # Pasul 1: Aplică transformările `after_item` (ex: Resize) din DataLoaders
        processed_item_pil = _learner.dls.after_item(img_pil_rgb)
        img_tensor = None

        # Pasul 2: Conversie la PyTorch Tensor
        if isinstance(processed_item_pil, torch.Tensor):
            img_tensor = processed_item_pil
        elif isinstance(processed_item_pil, (Image.Image, PILImage)):
            # Încercarea 1: ToTensor (fastai)
            try:
                candidate_fastai = fv_core.ToTensor()(processed_item_pil)
                if isinstance(candidate_fastai, torch.Tensor):
                    img_tensor = candidate_fastai
            except Exception: 
                pass

            # Încercarea 2: ToTensor (torchvision)
            if img_tensor is None:
                try:
                    pil_img_for_tv = processed_item_pil
                    # Se asigură că processed_item_pil este standard PIL.Image 
                    if not isinstance(processed_item_pil, Image.Image): 
                         if hasattr(processed_item_pil, 'convert'):
                            pil_img_for_tv = Image.fromarray(np.array(processed_item_pil.convert("RGB")))
                    
                    candidate_torchvision = TF.to_tensor(pil_img_for_tv)
                    if isinstance(candidate_torchvision, torch.Tensor):
                        img_tensor = candidate_torchvision
                except Exception: 
                    pass 
        else:
            st.error(f"XAI Error: Unexpected type for 'processed_item_pil': {type(processed_item_pil)}")
            return _pil_image

        # Verificare finală că avem un tensor după încercările de conversie
        if not isinstance(img_tensor, torch.Tensor):
            st.error(f"XAI Error: Image could not be converted to a tensor. 'img_tensor' is currently {type(img_tensor)}.")
            return _pil_image

        # Pasul 3: Adaugă dimensiunea de batch și mută pe device
        device = model_device(_learner.model)
        img_tensor_batch = img_tensor.unsqueeze(0).to(device)
        x_batch_for_norm = TensorImage(img_tensor_batch.float()) # se asigură că e TensorImage pentru normalizare

        # Pasul 4: Aplică Normalizarea Explicit 
        try:
            norm_transform = Normalize.from_stats(*imagenet_stats)
            img_tensor_batch_normalized = norm_transform(x_batch_for_norm)
        except Exception as e_explicit_norm:
            st.error(f"Eroare la aplicarea normalizării directe cu imagenet_stats: {e_explicit_norm}. Vizualizarea (Grad-CAM) poate fi incorect.")
            img_tensor_batch_normalized = x_batch_for_norm 

        # Layer-ul țintă pentru Grad-CAM 
        target_layer = None
        try: 
            if hasattr(_learner.model, 'layer4'): 
                target_layer = _learner.model.layer4[-1]
            elif hasattr(_learner.model, '0') and hasattr(_learner.model[0], '_modules'): 
                body_modules = list(_learner.model[0].children())
                for i in range(len(body_modules) - 1, -1, -1):
                    module = body_modules[i]
                    if isinstance(module, torch.nn.Sequential) and len(list(module.children())) > 0:
                        last_submodule_in_block = list(module.children())[-1]
                        if any(isinstance(m, torch.nn.Conv2d) for m in last_submodule_in_block.modules()):
                            target_layer = last_submodule_in_block
                            break
                        elif isinstance(last_submodule_in_block, torch.nn.Conv2d):
                            target_layer = last_submodule_in_block
                            break
                    elif any(isinstance(m, torch.nn.Conv2d) for m in module.modules()):
                         target_layer = module
                         break
            if target_layer is None: 
                potential_layers = [m for m in _learner.model.modules() if isinstance(m, torch.nn.Conv2d)]
                if potential_layers: target_layer = potential_layers[-1]
        except Exception as e_layer_sel:
            potential_layers = [m for m in _learner.model.modules() if isinstance(m, torch.nn.Conv2d)]
            if potential_layers: target_layer = potential_layers[-1]

        if target_layer is None:
            st.warning(TEXT_RO_XAI["xai_placeholder"] + " (Layer țintă invalid pentru vizualizare(Grad-CAM))")
            return _pil_image
        
        grad_cam_instance = ManualGradCAM(_learner.model, target_layer)
        heatmap_np = grad_cam_instance(img_tensor_batch_normalized, index=predicted_class_idx)
        grad_cam_instance.remove_hooks()

        if heatmap_np is None:
            st.info(TEXT_RO_XAI["xai_placeholder"] + " (heatmap_np este None)")
            return _pil_image

        heatmap_resized = cv2.resize(heatmap_np, (img_pil_rgb.width, img_pil_rgb.height)) # img_pil_rgb folosit pentru dimensiuni inițiale
        min_h, max_h = np.min(heatmap_resized), np.max(heatmap_resized)
        if max_h == min_h or (max_h - min_h < 1e-8):
            heatmap_normalized = np.zeros_like(heatmap_resized, dtype=np.float32)
        else:
            heatmap_normalized = (heatmap_resized - min_h) / (max_h - min_h)

        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_normalized), cv2.COLORMAP_JET)
        heatmap_colored_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        # original_img_np = np.array(img_pil_rgb)
        # superimposed_img_np = cv2.addWeighted(original_img_np, 0.6, heatmap_colored_rgb, 0.4, 0)
    
        # 1. Obține harta de contururi în loc de imaginea originală
        edge_map_np = get_edge_map(img_pil_rgb)

        # 2. Suprapune heatmap-ul peste harta de contururi
        #    Folosim 'edge_map_np' ca fundal. Parametrii de ponderare (0.6, 0.4) pot fi ajustați.
        superimposed_img_np = cv2.addWeighted(edge_map_np, 0.6, heatmap_colored_rgb, 0.4, 0)
        
        return Image.fromarray(superimposed_img_np)

    except Exception as e: 
        st.error(f"Eroare majoră la generarea Grad-CAM: {e}")
        import traceback
        traceback.print_exc()
        st.info(TEXT_RO_XAI["xai_placeholder"])
        return _pil_image
    
def generate_library_cam_visualization(_learner, _pil_image, predicted_class_idx, cam_method_class, use_edge_map=True):
    """Generează o vizualizare de tip CAM (ex: GradCAM) și o suprapune."""
    
    input_tensor, original_pil = preprocess_image(_learner, _pil_image)
    target_layer = find_target_layer(_learner)
    if target_layer is None:
        st.warning("Nu s-a putut identifica un strat țintă valid.")
        return None

    targets = [ClassifierOutputTarget(predicted_class_idx)]
    
    # Metodele care nu au nevoie de `targets`
    gradient_free_methods = [EigenCAM]

    grayscale_cam = None
    
    cam_instance = cam_method_class(model=_learner.model, target_layers=[target_layer])

    # Apelează metoda CAM
    if cam_method_class in gradient_free_methods:
        grayscale_cam = cam_instance(input_tensor=input_tensor, targets=None)
    else:
        grayscale_cam = cam_instance(input_tensor=input_tensor, targets=targets)
    
    if grayscale_cam is None: return None
    grayscale_cam = grayscale_cam[0, :]

    # Redimensionare și suprapunere
    background_img_np = np.array(original_pil) / 255.0
    if use_edge_map:
        background_img_np = get_edge_map(original_pil) / 255.0
        
    h, w, _ = background_img_np.shape
    grayscale_cam_resized = cv2.resize(grayscale_cam, (w, h))
    
    visualization = show_cam_on_image(background_img_np, grayscale_cam_resized, use_rgb=True)
    return Image.fromarray(visualization)

def generate_guided_backprop_image(_learner, _pil_image, predicted_class_idx):
    """Generează o vizualizare de tip Guided Backpropagation."""
    
    input_tensor, _ = preprocess_image(_learner, _pil_image)
    device = next(_learner.model.parameters()).device
    
    gb_model = GuidedBackpropReLUModel(model=_learner.model, device=device)
    gb = gb_model(input_tensor, target_category=predicted_class_idx)
    
    h, w, _ = np.array(_pil_image).shape
    
    gb_resized = cv2.resize(gb, (w, h))

    # normalizează imaginea pentru a fi vizibilă corect
    return Image.fromarray(deprocess_image(gb_resized))

def generate_guided_cam(_learner, _pil_image, predicted_class_idx, cam_method_class):
    """Combină Guided Backprop cu o metodă CAM specificată (manuală sau din librărie)."""
    input_tensor, original_pil = preprocess_image(_learner, _pil_image)
    target_layer = find_target_layer(_learner)
    if not target_layer: return None

    # --- Pasul 1: Generează Guided Backpropagation (comun pentru toate) ---
    device = next(_learner.model.parameters()).device
    gb_model = GuidedBackpropReLUModel(model=_learner.model, device=device)
    gb = gb_model(input_tensor, target_category=predicted_class_idx)

    # --- Pasul 2: Generează heatmap-ul în funcție de metoda aleasă ---
    grayscale_cam = None
    if cam_method_class == ManualGradCAM:
        # Folosește implementarea manuală
        cam_instance = ManualGradCAM(_learner.model, target_layer)
        grayscale_cam = cam_instance(input_tensor, index=predicted_class_idx)
        cam_instance.remove_hooks()
    else:
        # Folosește o metodă din bibliotecă
        cam_instance = cam_method_class(model=_learner.model, target_layers=[target_layer])
        targets = [ClassifierOutputTarget(predicted_class_idx)]
        result = cam_instance(input_tensor=input_tensor, targets=targets if cam_method_class != EigenCAM else None)
        if result is not None:
            grayscale_cam = result[0, :]

    if grayscale_cam is None: return None

    # --- Pasul 3: Combină cele două rezultate ---
    h, w = original_pil.height, original_pil.width
    # Redimensionează CAM-ul la dimensiunea imaginii originale
    cam_mask = cv2.resize(grayscale_cam, (w, h))
    cam_mask = cv2.merge([cam_mask, cam_mask, cam_mask])

    # Redimensionează și Guided Backprop (gb) la aceeași dimensiune
    gb_resized = cv2.resize(gb, (w, h))
    
    cam_gb = cam_mask * gb_resized
    # Face rezultatul vizibil

    return Image.fromarray(deprocess_image(cam_gb))

