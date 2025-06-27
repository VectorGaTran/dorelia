import streamlit as st
from PIL import Image, ImageOps
import io
import random
import json
from pathlib import Path
import torch

from db_utils import (
    get_gallery_from_db, remove_from_gallery_db, save_image, add_to_gallery_db,
    get_map_locations_from_db, remove_from_map_db, add_to_map_db,
    add_feedback_db, get_style_details
)

from xai_utils import generate_manual_gradcam_image, TEXT_RO_XAI, ManualGradCAM
from xai_utils import generate_library_cam_visualization, generate_guided_cam, generate_guided_backprop_image, LIBRARY_CAM_METHODS

import pandas as pd
import folium
from streamlit_folium import st_folium
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable
import base64

@st.cache_data(show_spinner=False)
def load_and_process_image(uploaded_file_bytes):
    img = Image.open(io.BytesIO(uploaded_file_bytes))
    img = ImageOps.exif_transpose(img)
    img = img.convert("RGB")
    img.thumbnail((1024, 1024))
    return img

def main_app_page(st_session_state, current_learner, TEXT_RO, KNOWN_ARCHITECTURAL_STYLES_list):
    st.title(TEXT_RO["app_title"])
    st.markdown("### (Detector Orientat spre Recunoașterea Esteticii Latente și Identificare Arhitecturală)")
    st.markdown(TEXT_RO["welcome_message"])
    st.markdown(TEXT_RO["app_description"])
    st.markdown("---")

    col1, col2 = st.columns([2, 3])

    with col1:
        st.subheader(TEXT_RO["upload_header"])
        uploaded_file = st.file_uploader(TEXT_RO["upload_prompt"], type=["jpg", "jpeg", "png"], key="file_uploader_main_ui")

        if uploaded_file is not None:
            if st_session_state.current_filename != uploaded_file.name or not st_session_state.current_pil_image: # Also check if image is not loaded
                st_session_state.current_predictions = None
                st_session_state.current_pil_image = None
                st_session_state.current_image_bytes_for_display = None
                st_session_state.current_filename = uploaded_file.name
                st_session_state.current_image_id_in_db = None
                st_session_state.feedback_submitted_for_current_image = False
            try:
                # image_pil = Image.open(uploaded_file)
                # image_pil = ImageOps.exif_transpose(image_pil)
                # image_pil_rgb = image_pil.convert('RGB')
                # acum e cached cu load and process image
                img_byte_arr = io.BytesIO()
                image_pil_rgb = load_and_process_image(uploaded_file.getvalue())
                image_pil_rgb.save(img_byte_arr, format='PNG')
                st_session_state.current_image_bytes_for_display = img_byte_arr.getvalue()
                #image_pil_rgb.thumbnail((1024, 1024)) 
                st.image(image_pil_rgb, caption=TEXT_RO["uploaded_image_caption"])
                st_session_state.current_pil_image = image_pil_rgb
            except Exception as e:
                st.error(f"{TEXT_RO['error_processing_image']} {e}")
                st_session_state.current_pil_image = None
                st_session_state.current_predictions = None
                st_session_state.current_filename = None
                st_session_state.current_image_bytes_for_display = None
                st_session_state.current_image_id_in_db = None
                st_session_state.feedback_submitted_for_current_image = False
                return
        else:
            if st_session_state.current_pil_image is not None or st_session_state.current_filename is not None:
                st_session_state.current_pil_image = None
                st_session_state.current_predictions = None
                st_session_state.current_filename = None
                st_session_state.current_image_bytes_for_display = None
                st_session_state.current_image_id_in_db = None
                st_session_state.feedback_submitted_for_current_image = False
                st.rerun() 
            return

    with col2:
        st.subheader(TEXT_RO["analysis_header"])
        if uploaded_file and st_session_state.current_pil_image is not None:
            button_key_suffix = f"{st_session_state.username}_{st_session_state.current_filename}" if st_session_state.current_filename else (uploaded_file.file_id if hasattr(uploaded_file, 'file_id') else str(random.randint(0,100000)))


            if st.button(TEXT_RO["analyze_button"], type="primary", key=f"analyze_btn_{button_key_suffix}"):
                if current_learner is None:
                    st.error("Modelul nu este încărcat. Vă rugăm selectați un model valid din bara laterală și reîncercați.")
                else:
                    with st.spinner(TEXT_RO["analyzing_spinner"]):
                        try:
                            pil_image_for_predict = st_session_state.current_pil_image
                            pred_class_str, pred_idx_tensor, outputs_tensor = current_learner.predict(pil_image_for_predict)
                            # se afișează maxim 5 predicții
                            k = min(5, len(current_learner.dls.vocab))
                            top_k_probs, top_k_indices = torch.topk(outputs_tensor, k=k)

                            predictions_list = []
                            for i in range(k):
                                idx_item = top_k_indices[i].item()
                                prob_item = top_k_probs[i].item()
                                style_name_item = current_learner.dls.vocab[idx_item]
                                query_style = style_name_item.split('(')[0].strip().replace(' ', '_')
                                details_link_item = f"https://fallingfalling.com"

                                style_details_db = get_style_details(style_name_item)
                                period_info = TEXT_RO.get("period_info_default", "Perioadă Est.: Necunoscută")
                                region_info = TEXT_RO.get("region_info_default", "Regiune: Necunoscută")
                                more_info_text_db = "Mai multe informații nu sunt disponibile."
                                if style_details_db:
                                    period_info = style_details_db.get("period", period_info)
                                    region_info = style_details_db.get("region", region_info)
                                    more_info_text_db = style_details_db.get("details_text", more_info_text_db)
                                    details_link_item = style_details_db.get("details_link", details_link_item)

                                predictions_list.append({
                                    "style": style_name_item, "confidence": prob_item,
                                    "link": details_link_item,
                                    "pred_idx": idx_item,
                                    "period": period_info,
                                    "region": region_info,
                                    "more_info_text": more_info_text_db
                                })
                            st_session_state.current_predictions = predictions_list
                            st_session_state.feedback_submitted_for_current_image = False
                            st_session_state.current_image_id_in_db = None 
                        except Exception as e_predict:
                            st.error(f"Eroare în timpul predicției: {e_predict}")
                            st_session_state.current_predictions = None

            if st_session_state.current_predictions:
                predictions = st_session_state.current_predictions
                st.success(TEXT_RO["analysis_complete"])

                primary_display_style = TEXT_RO["style_neidentificat"]
                display_confidence_message = TEXT_RO["confidence_low_message"]
                show_xai_for = []

                confidences_only = [p['confidence'] for p in predictions]
                styles_only = [p['style'] for p in predictions]
                pred_indices_only = [p['pred_idx'] for p in predictions]

                high_conf_predictions = [(s, c, pi) for s, c, pi in zip(styles_only, confidences_only, pred_indices_only) if c >= 0.40]
                high_conf_predictions.sort(key=lambda x: x[1], reverse=True)

                if not high_conf_predictions:
                    primary_display_style = TEXT_RO["style_neidentificat"]
                    display_confidence_message = TEXT_RO["confidence_low_message"]
                elif len(high_conf_predictions) == 1:
                    primary_display_style = high_conf_predictions[0][0]
                    display_confidence_message = TEXT_RO["high_confidence_message"]
                    if high_conf_predictions[0][1] >= 0.40:
                        show_xai_for.append(high_conf_predictions[0])
                else:
                    style1_hc, conf1_hc, p_idx1_hc = high_conf_predictions[0]
                    style2_hc, conf2_hc, p_idx2_hc = high_conf_predictions[1]
                    primary_display_style = TEXT_RO["style_amestec"].format(style1=style1_hc, style2=style2_hc)
                    display_confidence_message = TEXT_RO["mixed_style_message"]
                    if conf1_hc >= 0.40: show_xai_for.append(high_conf_predictions[0])
                    if conf2_hc >= 0.40: show_xai_for.append(high_conf_predictions[1])

                st.markdown(f"### **{primary_display_style}**")
                st.info(display_confidence_message)
                
                if show_xai_for:
                    st.subheader("Analiză XAI (AI Explicabil)")

                    # --- Selectarea tipului general de vizualizare ---
                    viz_type_options = ["Heatmap", "Guided Backpropagation", "Guided CAM (Combinat)"]
                    viz_type_key = st.selectbox(
                        "Alege tipul de vizualizare:",
                        options=viz_type_options,
                        key=f"viz_type_select_{button_key_suffix}" # Cheie unică pentru a evita probleme de stare
                    )

                    # --- Selectarea metodei specifice de CAM ---
                    cam_method_class = None
                    # Afișăm acest meniu doar dacă vizualizarea necesită o metodă CAM
                    if viz_type_key in ["Heatmap", "Guided CAM (Combinat)"]:
                        # Creează o listă completă de opțiuni, cu cea manuală prima
                        full_cam_options = {"Grad-CAM (Manual)": ManualGradCAM, **LIBRARY_CAM_METHODS}
                        
                        selected_cam_name = st.selectbox(
                            "Alege metoda CAM specifică:",
                            options=list(full_cam_options.keys()),
                            key=f"cam_method_select_{button_key_suffix}" # Cheie unică
                        )
                        cam_method_class = full_cam_options[selected_cam_name]

                    # --- Butonul de generare ---
                    button_label = f"Generează: {selected_cam_name}" if cam_method_class else f"Generează: {viz_type_key}"
                    if st.button(button_label, key=f"generate_button_{button_key_suffix}"):
                        
                        # Iterează prin fiecare predicție de top (aici se obține pred_idx_xai)
                        for style_xai, conf_xai, pred_idx_xai in show_xai_for:
                            
                            expander_title = f"Explicație pentru: {style_xai} ({conf_xai*100:.1f}%)"
                            with st.expander(expander_title, expanded=True):
                                
                                spinner_text = f"Se generează vizualizarea cu {selected_cam_name if cam_method_class else viz_type_key}..."
                                with st.spinner(spinner_text):
                                    xai_image = None
                                    
                                    # --- Logica de apelare a funcției corecte cu toți argumentele ---
                                    
                                    if viz_type_key == "Heatmap":
                                        if cam_method_class == ManualGradCAM:
                                            xai_image = generate_manual_gradcam_image(
                                                _learner=current_learner,
                                                _pil_image=st_session_state.current_pil_image,
                                                predicted_class_idx=pred_idx_xai,
                                                image_identifier_for_cache=f"{st_session_state.current_filename}_{pred_idx_xai}"
                                            )
                                        else:
                                            xai_image = generate_library_cam_visualization(
                                                _learner=current_learner,
                                                _pil_image=st_session_state.current_pil_image,
                                                predicted_class_idx=pred_idx_xai,
                                                cam_method_class=cam_method_class
                                            )
                                    
                                    elif viz_type_key == "Guided Backpropagation":
                                        xai_image = generate_guided_backprop_image(
                                            _learner=current_learner,
                                            _pil_image=st_session_state.current_pil_image,
                                            predicted_class_idx=pred_idx_xai
                                        )
                                    
                                    elif viz_type_key == "Guided CAM (Combinat)":
                                        xai_image = generate_guided_cam(
                                            _learner=current_learner,
                                            _pil_image=st_session_state.current_pil_image,
                                            predicted_class_idx=pred_idx_xai,
                                            cam_method_class=cam_method_class
                                        )

                                    # --- Afișarea rezultatului ---
                                    if xai_image:
                                        caption_text = f"Rezultat pentru {selected_cam_name if cam_method_class else viz_type_key}"
                                        st.image(xai_image, caption=caption_text)
                                    else:
                                        st.warning("Vizualizarea nu a putut fi generată.")

                st.markdown("---")


                show_predictions = st.checkbox("Afișează top predicții detaliate")
                if show_predictions and predictions:
                    st.markdown("#### Top Predicții Detaliate:")
                    for i, pred_info in enumerate(predictions):
                        style = pred_info["style"]
                        confidence = pred_info["confidence"]
                        period = pred_info.get("period", TEXT_RO.get("period_info_default", "Perioadă Est.: Necunoscută"))
                        region = pred_info.get("region", TEXT_RO.get("region_info_default", "Regiune: Necunoscută"))
                        more_info_text = pred_info.get("more_info_text", "Detalii indisponibile.")
                        details_link = pred_info.get("link", "#")

                        st.markdown(f"**{i+1}. {style}**")
                        st.progress(int(confidence * 100))
                        st.markdown(f"**{TEXT_RO['prediction_confidence']}:** {confidence*100:.2f}%")
                        st.markdown(f"*{period}* | *{region}*")

                        with st.expander(f"{TEXT_RO['more_info_link'].format(style=style)}"):
                            st.markdown(more_info_text)
                            st.markdown(f"[Link Info]({details_link})", unsafe_allow_html=True)
                        st.markdown("---")

                if not st_session_state.feedback_submitted_for_current_image:
                    st.subheader(TEXT_RO["feedback_header"])
                    feedback_form_key = f"feedback_form_{button_key_suffix}"
                    feedback_form = st.form(key=feedback_form_key)
                    with feedback_form:
                        st.write(TEXT_RO["feedback_prompt"])
                        feedback_cols_buttons = st.columns(2)
                        correct_prediction = feedback_cols_buttons[0].form_submit_button(TEXT_RO["feedback_yes"])
                        incorrect_prediction_btn = feedback_cols_buttons[1].form_submit_button(TEXT_RO["feedback_no"])

                        current_suggestion_box_id = f"suggestion_for_{button_key_suffix}"
                        if 'show_suggestion_box_for_id' not in st_session_state or st_session_state.show_suggestion_box_for_id != current_suggestion_box_id:
                            st_session_state.show_suggestion_box_active = False

                        if incorrect_prediction_btn:
                            st_session_state.show_suggestion_box_active = True
                            st_session_state.show_suggestion_box_for_id = current_suggestion_box_id

                        if st_session_state.get('show_suggestion_box_active', False) and st_session_state.get('show_suggestion_box_for_id') == current_suggestion_box_id:
                            suggested_style_input = st.text_input(TEXT_RO["feedback_suggest_style_prompt"], key=f"suggest_{button_key_suffix}")
                            if st.form_submit_button(TEXT_RO["feedback_submit_button"]):
                                add_feedback_db(st_session_state.username, st_session_state.current_image_id_in_db, st_session_state.current_filename, primary_display_style, False, suggested_style_input if suggested_style_input else None)
                                st.toast(TEXT_RO["feedback_thanks_no"])
                                st_session_state.feedback_submitted_for_current_image = True
                                st_session_state.show_suggestion_box_active = False
                                st.rerun()

                        if correct_prediction:
                            add_feedback_db(st_session_state.username, st_session_state.current_image_id_in_db, st_session_state.current_filename, primary_display_style, True, None)
                            st.toast(TEXT_RO["feedback_thanks_yes"])
                            st_session_state.feedback_submitted_for_current_image = True
                            st_session_state.show_suggestion_box_active = False 
                            st.rerun()
                        
                        if incorrect_prediction_btn and not st_session_state.get('show_suggestion_box_active', False) :
                            st_session_state.show_suggestion_box_active = True
                            st_session_state.show_suggestion_box_for_id = current_suggestion_box_id
                            st.rerun() 

                    st.caption(TEXT_RO["feedback_caption"])
                else:
                    st.success("Feedback-ul pentru această imagine a fost trimis.")
                st.markdown("---")

                if st_session_state.current_image_bytes_for_display:
                    st.subheader(TEXT_RO["save_locate_header"])
                    unique_key_suffix_save = f"{st_session_state.username}_{st_session_state.current_filename}" if st_session_state.current_filename else f"{st_session_state.username}_save_area"

                    user_notes = st.text_area(TEXT_RO["notes_prompt"], key=f"notes_{unique_key_suffix_save}")
                    address_input = st.text_input(TEXT_RO["address_prompt"], key=f"address_{unique_key_suffix_save}")

                    col_save1, col_save2 = st.columns(2)
                    with col_save1:
                        if st.button(TEXT_RO["add_gallery_button"], key=f"add_gallery_btn_{unique_key_suffix_save}"):
                            image_path = save_image(st_session_state.current_image_bytes_for_display, st_session_state.current_filename, st_session_state.username)
                            if image_path:
                                # Se verifică dacă imaginea nu există deja pentru a evita duplicatele.
                                gallery_items_db = get_gallery_from_db(st_session_state.username, text_ro_config=TEXT_RO)
                                is_duplicate_db = any(item['filename'] == st_session_state.current_filename  for item in gallery_items_db)

                                if not is_duplicate_db:
                                    image_db_id = add_to_gallery_db(st_session_state.username, st_session_state.current_filename, image_path,
                                                                    st_session_state.current_predictions, user_notes, address_input if address_input else None, TEXT_RO)
                                    if image_db_id: 
                                        st_session_state.current_image_id_in_db = image_db_id 
                                        st.success(TEXT_RO["gallery_added_success"].format(filename=st_session_state.current_filename))
                                        st.session_state.image_added_to_gallery = True
                                    else: 
                                        st.warning(TEXT_RO["gallery_already_exists"].format(filename=st_session_state.current_filename))
                                        st.session_state.image_added_to_gallery = False

                                else:
                                    st.warning(TEXT_RO["gallery_already_exists"].format(filename=st_session_state.current_filename))
                                    st.session_state.image_added_to_gallery = False

                            else:
                                st.error("Nu s-a putut salva imaginea pentru galerie.")

                    with col_save2:
                        if st.button(TEXT_RO["add_map_button"], key=f"add_map_btn_{unique_key_suffix_save}"):
                                if not st.session_state.get("image_added_to_gallery", False):
                                    st.error("Trebuie să adăugați imaginea în galerie înainte de a o adăuga pe hartă.")
                                elif address_input:
                                    try:
                                        geolocator_agent = f"app_arhitectura_{st_session_state.username}_{random.randint(0,10000)}"
                                        geolocator = Nominatim(user_agent=geolocator_agent)
                                        with st.spinner(f"Se geocodează '{address_input}'..."):
                                            location = geolocator.geocode(address_input, timeout=10)

                                        if location:
                                            lat, lon = location.latitude, location.longitude
                                            map_locations_db = get_map_locations_from_db(st_session_state.username, text_ro_config=TEXT_RO)
                                            is_duplicate_map_db = any(item['address'] == address_input and item['filename'] == st_session_state.current_filename for item in map_locations_db)

                                            if not is_duplicate_map_db:
                                                thumb_img = Image.open(io.BytesIO(st_session_state.current_image_bytes_for_display))
                                                thumb_img.thumbnail((150, 150))
                                                thumb_byte_arr = io.BytesIO()
                                                thumb_img.save(thumb_byte_arr, format='PNG')
                                                image_preview_path = save_image(thumb_byte_arr.getvalue(), f"thumb_{st_session_state.current_filename}", st_session_state.username, is_thumbnail=True)

                                                if image_preview_path:
                                                    map_display_style = primary_display_style if primary_display_style else (predictions[0]['style'] if predictions else TEXT_RO["style_neidentificat"])
                                                    add_to_map_db(st_session_state.username, address_input, lat, lon,
                                                                map_display_style,
                                                                image_preview_path,
                                                                st_session_state.current_filename)
                                                    st.success(TEXT_RO["map_added_success"].format(address=address_input, lat=lat, lon=lon))
                                                else:
                                                    st.error("Nu s-a putut salva thumbnai; pentru hartă.")
                                            else:
                                                st.warning(TEXT_RO["gallery_already_exists"].format(filename=f"{st_session_state.current_filename} la adresa {address_input}")) # Clarify warning
                                        else:
                                            st.error(TEXT_RO["map_address_not_found"].format(address=address_input))
                                    except ImportError:
                                        st.error("Biblioteca Geopy nu este instalată. Rulați `pip install geopy`.")
                                    except GeocoderTimedOut:
                                        st.error(TEXT_RO["map_geocoding_error"].format(e="Timeout"))
                                    except GeocoderUnavailable:
                                        st.error(TEXT_RO["map_geocoding_error"].format(e="Serviciu indisponibil"))
                                    except Exception as e_geo:
                                        st.error(TEXT_RO["map_unexpected_geocoding_error"].format(e=e_geo))
                                else:
                                    st.warning(TEXT_RO["map_enter_address_warning"])
            elif uploaded_file and not st_session_state.current_predictions :
                st.info("Apăsați pe 'Analizează Stilul' pentru a vedea rezultatele.")


def gallery_page(st_session_state, TEXT_RO):
    st.title(TEXT_RO["gallery_title"])
    st.markdown(TEXT_RO["gallery_description"])

    
    gallery_items = get_gallery_from_db(st_session_state.username, text_ro_config=TEXT_RO)

    if not gallery_items:
        st.info(TEXT_RO["gallery_empty"])
        return

    all_styles_in_gallery = sorted(list(set(item['primary_style'] for item in gallery_items if item['primary_style'])))

    filter_options = [TEXT_RO["all_styles_option"]]
    neidentificat_label = TEXT_RO.get("style_neidentificat", "Neidentificat")
    amestec_prefix = TEXT_RO.get("style_amestec", "Amestec {style1} + {style2}").split(" ")[0]

    if neidentificat_label in all_styles_in_gallery:
        filter_options.append(TEXT_RO.get("neidentificat_option", "Neidentificat")) 
    if any(style.startswith(amestec_prefix) for style in all_styles_in_gallery if style):
        filter_options.append(TEXT_RO.get("amestec_option", "Amestec de Stiluri"))

    specific_styles_in_gallery = [s for s in all_styles_in_gallery if s and s != neidentificat_label and not s.startswith(amestec_prefix)]
    filter_options.extend(sorted(specific_styles_in_gallery))

    selected_style_filter = TEXT_RO["all_styles_option"] 
    if all_styles_in_gallery: 
        selected_style_filter = st.selectbox(
            TEXT_RO["filter_gallery_prompt"],
            options=filter_options,
            key=f"gallery_filter_db_ui_{st_session_state.username}" 
        )

    if selected_style_filter == TEXT_RO["all_styles_option"]:
        filtered_gallery = gallery_items
    elif selected_style_filter == TEXT_RO.get("neidentificat_option", "Neidentificat"):
        filtered_gallery = [item for item in gallery_items if item['primary_style'] == neidentificat_label]
    elif selected_style_filter == TEXT_RO.get("amestec_option", "Amestec de Stiluri"):
        filtered_gallery = [item for item in gallery_items if item['primary_style'] and item['primary_style'].startswith(amestec_prefix)]
    else:
        filtered_gallery = [item for item in gallery_items if item['primary_style'] == selected_style_filter]

    if not filtered_gallery:
        st.warning(TEXT_RO["no_images_for_style"].format(style=selected_style_filter))
        return

    num_columns = st.slider(TEXT_RO["gallery_columns_slider"], 1, 5, 3, key=f"gallery_cols_db_ui_{st_session_state.username}")

    cols = st.columns(num_columns)
    for index, item in enumerate(filtered_gallery):
        col_index = index % num_columns
        with cols[col_index]:
            if item['image_bytes']:
                st.image(item['image_bytes'], caption=f"Fișier: {item.get('filename', 'N/A')}")
            else:
                st.warning(f"Imaginea pentru {item.get('filename', 'N/A')} nu a putut fi încărcată.")

            primary_style_gal = item['primary_style']
            primary_confidence_gal = item['primary_confidence']

            if primary_style_gal == neidentificat_label:
                 st.markdown(f"**{TEXT_RO['gallery_top_style']}:** {primary_style_gal}")
            elif primary_style_gal and primary_style_gal.startswith(amestec_prefix):
                 st.markdown(f"**{TEXT_RO['gallery_top_style']}:** {primary_style_gal}")
            else:
                 st.markdown(f"**{TEXT_RO['gallery_top_style']}:** {primary_style_gal} ({primary_confidence_gal*100:.1f}%)")

            expander_key_gallery = f"gallery_expander_{item['id']}_{st_session_state.username}"
            with st.expander(TEXT_RO["gallery_view_analysis_notes"]):
                num_preds_to_show_gallery = min(3, len(item['predictions']))
                for i, pred_info in enumerate(item['predictions'][:num_preds_to_show_gallery]):
                    style_det = pred_info['style']
                    confidence_det = pred_info['confidence']
                    link_det = pred_info.get('link', '#')
                    period_det = pred_info.get('period', TEXT_RO.get("period_info_default", "Perioadă Est.: Necunoscută"))
                    region_det = pred_info.get('region', TEXT_RO.get("region_info_default", "Regiune: Necunoscută"))
                    st.markdown(f"**{i+1}. {style_det}** ({confidence_det*100:.1f}%)")
                    st.markdown(f"   *{period_det} | {region_det}*")
                    st.markdown(f"[Detalii Wikipedia]({link_det})", unsafe_allow_html=True)
                    if i < num_preds_to_show_gallery - 1:
                        st.markdown("---")

                if item['notes']:
                    st.markdown(f"**{TEXT_RO['gallery_your_notes']}**")
                    st.info(item['notes'])
                if item['address']:
                    st.markdown(f"**{TEXT_RO['gallery_saved_address']}** {item['address']}")

            remove_btn_key_gallery_ui = f"remove_gallery_db_ui_{item['id']}_{st_session_state.username}"
            if st.button(TEXT_RO["gallery_remove_button"].format(filename=item.get('filename', 'Imagine')[:15]), key=remove_btn_key_gallery_ui):
                
                if remove_from_gallery_db(item['id'], st_session_state.username):
                    st.success(f"'{item.get('filename', 'Imagine')}' a fost șters/ștearsă din galerie.")
                    st.rerun()
                else:
                    st.error(TEXT_RO["gallery_remove_error"])
            st.markdown("---")


def map_favorites_page(st_session_state, TEXT_RO):
    st.title(TEXT_RO["map_title"])
    st.markdown(TEXT_RO["map_description"])

    
    map_locations = get_map_locations_from_db(st_session_state.username, text_ro_config=TEXT_RO)

    if not map_locations:
        st.info(TEXT_RO["map_empty"])
        return

    try:
        df_locations = pd.DataFrame(map_locations)
        if df_locations.empty: 
            st.info(TEXT_RO["map_empty"])
            return

        all_styles_on_map = sorted(list(set(item['style'] for item in map_locations if item['style'])))
        neidentificat_label_map = TEXT_RO.get("style_neidentificat", "Neidentificat")
        amestec_prefix_map = TEXT_RO.get("style_amestec", "Amestec {style1} + {style2}").split(" ")[0]

        filter_options_map = [TEXT_RO["all_styles_option"]]
        if neidentificat_label_map in all_styles_on_map:
            filter_options_map.append(TEXT_RO.get("neidentificat_option", "Neidentificat"))
        if any(style.startswith(amestec_prefix_map) for style in all_styles_on_map if style):
            filter_options_map.append(TEXT_RO.get("amestec_option", "Amestec de Stiluri"))

        specific_styles_on_map = [s for s in all_styles_on_map if s and s != neidentificat_label_map and not s.startswith(amestec_prefix_map)]
        filter_options_map.extend(sorted(specific_styles_on_map))

        selected_style_filter_map = TEXT_RO["all_styles_option"] 
        if all_styles_on_map: # arată selectbok numai dacă există stiluri
            selected_style_filter_map = st.selectbox(
                TEXT_RO["filter_map_prompt"],
                options=filter_options_map,
                key=f"map_filter_db_ui_{st_session_state.username}"
            )

        if selected_style_filter_map == TEXT_RO["all_styles_option"]:
            df_to_display = df_locations
        elif selected_style_filter_map == TEXT_RO.get("neidentificat_option", "Neidentificat"):
            df_to_display = df_locations[df_locations['style'] == neidentificat_label_map]
        elif selected_style_filter_map == TEXT_RO.get("amestec_option", "Amestec de Stiluri"):
             df_to_display = df_locations[df_locations['style'].str.startswith(amestec_prefix_map, na=False)]
        else:
            df_to_display = df_locations[df_locations['style'] == selected_style_filter_map]

        if df_to_display.empty:
            st.warning(TEXT_RO["no_locations_for_style"].format(style=selected_style_filter_map))
            return

        map_center = [df_to_display['latitude'].mean(), df_to_display['longitude'].mean()] if not df_to_display.empty else [45.9432, 24.9668] # Romania center
        m = folium.Map(location=map_center, zoom_start=7)

        for _, row in df_to_display.iterrows():
            popup_header = f"<b>Fișier:</b> {row['filename']}<br><b>Stil:</b> {row['style']}<br><b>Adresă:</b> {row['address']}"
            img_html_part = "<br>(Imagine lipsă)"

            if row.get('image_preview_bytes'): 
                img_data = base64.b64encode(row['image_preview_bytes']).decode('utf-8')
                img_html_part = f"""
                    <br>
                    <img src="data:image/png;base64,{img_data}" alt="Previzualizare" style="width:100px;height:auto;border:1px solid #ccc;">
                """

            popup_content = popup_header + img_html_part
            html_for_popup = f'<div style="width: 180px; height: auto; max-height: 200px; overflow-y: auto;">{popup_content}</div>'
            iframe = folium.IFrame(html_for_popup, width=200, height=220)
            popup = folium.Popup(iframe, max_width=260)

            folium.Marker(
                [row['latitude'], row['longitude']],
                popup=popup,
                tooltip=f"{row['filename']} ({row['style']})",
                icon=folium.Icon(color="blue", icon="info-sign", prefix="glyphicon")
            ).add_to(m)

        st_folium(m, width=725, height=500, key=f"folium_map_{st_session_state.username}") 

        st.subheader(TEXT_RO["map_manage_locations_header"])
        location_options_map_ui = {}
        if not df_to_display.empty:
            location_options_map_ui = {
                f"{item['filename']} @ {item['address']} (Stil: {item['style']})": item['id']
                for index, item in df_to_display.iterrows() # folosește df_to_display ce este deja filtrată
            }
        
        # afișează multiselect numai dacă există opțiuni
        if location_options_map_ui:
            locations_to_remove_display_texts = st.multiselect(
                TEXT_RO["map_remove_select_prompt"],
                options=list(location_options_map_ui.keys()),
                key=f"map_remove_select_db_ui_{st_session_state.username}" 
            )
            if st.button(TEXT_RO["map_remove_selected_button"], key=f"remove_map_button_db_ui_{st_session_state.username}"): 
                if not locations_to_remove_display_texts:
                    st.info(TEXT_RO["map_remove_none_selected"])
                else:
                    ids_to_remove = [location_options_map_ui[text] for text in locations_to_remove_display_texts if text in location_options_map_ui]
                    if ids_to_remove:
                        
                        removed_count = remove_from_map_db(ids_to_remove, st_session_state.username)
                        if removed_count > 0:
                            st.success(TEXT_RO["map_remove_success"].format(count=removed_count))
                            st.rerun()
                        else:
                            st.error(TEXT_RO["map_remove_error"]) 
                    else:
                        st.info(TEXT_RO["map_remove_none_selected"]) 
        else:
            st.info("Nu există locații pentru filtrul curent.")


    except ImportError:
        st.warning(TEXT_RO["map_libs_missing_warning"])
        st.image("https://placehold.co/800x400/E0E0E0/757575?text=Hartă+Interactivă+-+Necesită+Instalare", caption=TEXT_RO["map_libs_missing_image_caption"])
    except Exception as e_map:
        st.error(TEXT_RO["map_render_error"].format(e=e_map))
        import traceback
        traceback.print_exc()


def thesis_info_page(TEXT_RO): # informație statică
    st.title(TEXT_RO["about_title"])
    st.markdown("""
    Această aplicație este parte a unui proiect de licență axat pe **Recunoașterea Automată a Stilurilor Arhitecturale utilizând Deep Learning**.

    **Obiective Principale:**
    - Dezvoltarea unui model AI capabil să identifice stiluri arhitecturale din imagini.
    - Implementarea unui sistem ce oferă scoruri de încredere pentru predicțiile sale.
    - Explorarea metodelor de explicare a deciziilor AI (XAI) precum Grad-CAM pentru predicțiile cu încredere relevantă (>40%).
    - Integrarea unei abordări hibride ce combină recunoașterea automată cu feedback uman, cu stocarea feedback-ului pentru analize viitoare.
    - Analiza și etichetarea imaginilor ca "Neidentificat" (sub 40% încredere) sau "Amestec de stiluri" (două stiluri >40% încredere).
    - Persistența datelor utilizatorului (galerie, hartă, notițe, feedback) folosind o bază de date locală SQLite, specifică fiecărui utilizator autentificat.
    - Personalizarea informațiilor afișate pentru fiecare stil (perioadă, regiune, detalii suplimentare).

    **Tehnologii de Bază:**
    - **Python**, **Fastai**, **Streamlit**, **Pillow**, **Geopy**, **SQLite**.
    """)
    