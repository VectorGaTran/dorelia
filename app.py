import streamlit as st
from PIL import Image
from pathlib import Path
import platform
import multiprocessing
import pathlib
import os

import torch
from model_utils import load_learner_and_vocab
from db_utils import init_db, add_user_db, check_user_db
from ui_pages import main_app_page, gallery_page, map_favorites_page, thesis_info_page

st.set_page_config(
    page_title="Dorelia",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

from config import MODELS_DIR, DEFAULT_MODEL_NAME
KNOWN_ARCHITECTURAL_STYLES = []

# Dicționar de text ca aplicația să poată avea mai multe limbi disponibile
TEXT_RO = {
    "app_title": "DORELIA",
    "welcome_message": "Bine ați venit!",
    "app_description": """Această aplicație încearcă să identifice stilul arhitectural al clădirii din imagine (sau de către ce stiluri din listă a fost influențată arhitectura clădirii).
        Se recomandă o imagine care să aibă o singură clădire (sau cel puțin cea de se dorește analizată să fie în prim plan), iar imaginea să nu conțină prea multă vegetație vizibilă.""",
    "upload_header": "Încărcați Imaginea Clădirii",
    "upload_prompt": "Alegeți o imagine...",
    "uploaded_image_caption": "Imagine Încărcată.",
    "error_processing_image": "Eroare la deschiderea sau procesarea imaginii:",
    "info_upload_first": "Vă rugăm să încărcați mai întâi o imagine pentru a putea fi analizată.",
    "analysis_header": "Rezultatele Analizei",
    "analyze_button": "Analizează Stilul",
    "analyzing_spinner": "Se analizează imaginea...",
    "analysis_complete": "Analiză Completă!",
    "prediction_confidence": "Încredere",
    "more_info_link": "Detalii despre {style}",
    "xai_expander_title": "Explicația Caracteristicilor pentru {style} (XAI - Grad-CAM)",
    "xai_placeholder": "Vizualizarea Grad-CAM nu a putut fi generată.",
    "xai_generating": "Se generează Grad-CAM...",
    "feedback_header": "Feedback-ul tău",
    "feedback_prompt": "Ajutați la îmbunătățirea modelului! Credeți că predicția principală este corectă?",
    "feedback_yes": "Da, pare corect!",
    "feedback_no": "Nu, pare greșit.",
    "feedback_suggest_style_prompt": "Sugerați stilul corect (opțional):",
    "feedback_submit_button": "Trimite Feedback",
    "feedback_thanks_yes": "Mulțumim pentru confirmare!",
    "feedback_thanks_no": "Mulțumim pentru feedback! Sugestia ta a fost înregistrată.",
    "feedback_caption": "(Feedback-ul va ajuta la rafinarea viitoarelor versiuni ale modelului AI)",
    "save_locate_header": "Salvare și Localizare (pe hartă)",
    "notes_prompt": "Adăugați notițe personale (opțional):",
    "address_prompt": "Introduceți adresa pentru hartă (opțional, ex: 'Strada Pictor Arthur Verona 13-15, București'):",
    "add_gallery_button": "Adaugă la Galerie",
    "add_map_button": "Adaugă și pe Hartă ",
    "gallery_added_success": "'{filename}' a fost adăugat(ă) în galerie!",
    "gallery_already_exists": "'{filename}' este deja în galerie sau nu a putut fi salvată.",
    "map_added_success": "Locația '{address}' ({lat:.4f}, {lon:.4f}) a fost adăugată pe hartă!",
    "map_address_not_found": "Adresa '{address}' nu a fost găsită sau nu a putut fi geocodată.",
    "map_geocoding_error": "Eroare la serviciul de geocodare: {e}. Vă rugăm încercați mai târziu.",
    "map_unexpected_geocoding_error": "O eroare neașteptată a apărut în timpul geocodării: {e}",
    "map_enter_address_warning": "Vă rugăm introduceți o adresă pentru a o adăuga pe hartă.",
    "gallery_title": "Galeria Mea de Imagini Arhitecturale",
    "gallery_description": "Vizualizați și filtrați imaginile salvate după stilul arhitectural principal prezis.",
    "gallery_empty": "Galeria este goală. Analizați imagini și folosiți butonul 'Adaugă la Galerie'!",
    "filter_gallery_prompt": "Filtrează după Stilul Arhitectural Principal:",
    "all_styles_option": "Toate Stilurile",
    "neidentificat_option": "Neidentificat",
    "amestec_option": "Amestec de Stiluri",
    "no_images_for_style": "Nicio imagine găsită pentru stilul: {style}",
    "gallery_columns_slider": "Număr de coloane în galerie:",
    "gallery_top_style": "Stil Principal",
    "gallery_view_analysis_notes": "Vezi Analiza Completă & Notițe",
    "gallery_your_notes": "Notițele tale:",
    "gallery_saved_address": "Adresă Salvată:",
    "gallery_remove_button": "Șterge '{filename}'",
    "gallery_remove_error": "Eroare: Nu s-a putut găsi elementul specific pentru ștergere.",
    "map_title": "Harta Clădirilor Favorite",
    "map_description": "Vizualizați locațiile clădirilor salvate, filtrabile după stilul arhitectural.",
    "map_empty": "Harta este goală. Analizați imagini, introduceți o adresă și folosiți 'Adaugă și pe Hartă'!",
    "filter_map_prompt": "Filtrează harta după Stilul Arhitectural:",
    "no_locations_for_style": "Nicio locație găsită pentru stilul: {style}",
    "map_manage_locations_header": "Gestionează Locațiile de pe Hartă",
    "map_remove_select_prompt": "Selectați locațiile pentru a le șterge de pe hartă:",
    "map_remove_selected_button": "Șterge Locațiile Selectate de pe Hartă",
    "map_remove_none_selected": "Nicio locație selectată pentru ștergere.",
    "map_remove_success": "Au fost șterse {count} locații.",
    "map_remove_error": "Nicio locație potrivită nu a fost găsită pentru ștergere sau selecția a fost deja procesată.",
    "map_libs_missing_warning": "Funcționalitatea hărții necesită `pandas`. Vă rugăm instalați-le: `pip install pandas geopy`",
    "map_libs_missing_image_caption": "Instalați bibliotecile pentru hartă.",
    "map_render_error": "O eroare a apărut la redarea hărții: {e}",
    "map_data_format_error": "Vă rugăm asigurați-vă că datele de locație sunt formatate corect.",
    "about_title": "Despre Acest Proiect",
    "sidebar_navigation_header": "Navigație",
    "sidebar_goto_prompt": "Mergi la",
    "sidebar_info_title": "Proiect de Licență: Recunoașterea Stilurilor Arhitecturale",
    "sidebar_info_focus": "Accent: Stilul Neoromânesc",
    "sidebar_info_upload_instruction": "Încărcați o imagine în pagina 'Analizează Stilul'.",
    "footer_text": "Aplicație Recunoaștere Stiluri Arhitecturale - Proiect de Licență 2025",
    "style_neidentificat": "Neidentificat",
    "style_amestec": "Amestec {style1} + {style2}",
    "confidence_low_message": "Predicție cu încredere scăzută (<40%). Marcat ca 'Neidentificat'.",
    "mixed_style_message": "Două stiluri cu încredere ridicată (>40%). Marcat ca 'Amestec'.",
    "high_confidence_message": "Încredere ridicată.",
    "period_info_default": "Perioadă Est.: Necunoscută",
    "region_info_default": "Regiune: Necunoscută",
    "login_tab": "Autentificare",
    "signup_tab": "Înregistrare Cont Nou",
    "username_label": "Nume Utilizator:",
    "password_label": "Parolă:",
    "login_button": "Autentificare",
    "signup_button": "Înregistrează Cont",
    "logout_button": "Deconectare",
    "login_success": "Autentificare reușită!",
    "login_failed": "Nume de utilizator sau parolă incorectă.",
    "signup_failed_format": "Numele de utilizator și parola nu pot fi goale.",
    "signup_failed_exists": "Numele de utilizator există deja.",
    "signup_failed_general": "Eroare la înregistrare: {e}",
    "signup_success": "Utilizator înregistrat cu succes! Vă puteți autentifica acum.",
    "auth_page_title": "Autentificare Utilizator",
    "logged_in_as": "Autentificat ca:",
}

init_db()

# Inițializează session state pentru autentificare
if "username" not in st.session_state:
    st.session_state.username = None
# Verifică dacă utilizatorul este autentificat
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# Alte inițializări session state
if "current_model_path" not in st.session_state:
    st.session_state.current_model_path = None
if "learner" not in st.session_state:
    st.session_state.learner = None
if "current_pil_image" not in st.session_state:
    st.session_state.current_pil_image = None
if "current_image_bytes_for_display" not in st.session_state:
    st.session_state.current_image_bytes_for_display = None
if "current_predictions" not in st.session_state:
    st.session_state.current_predictions = None
if "current_filename" not in st.session_state:
    st.session_state.current_filename = None
if "current_image_id_in_db" not in st.session_state:
    st.session_state.current_image_id_in_db = None
if "feedback_submitted_for_current_image" not in st.session_state:
    st.session_state.feedback_submitted_for_current_image = False
if "show_suggestion_box" not in st.session_state:
    st.session_state.show_suggestion_box = False
if "show_suggestion_box_for" not in st.session_state:
    st.session_state.show_suggestion_box_for = None


def list_models(models_dir):
    if not models_dir.is_dir():
        return []
    return [f.name for f in models_dir.glob("*.pkl")]

def show_login_register_forms():
    st.title(TEXT_RO["auth_page_title"])
    login_tab, signup_tab = st.tabs([TEXT_RO["login_tab"], TEXT_RO["signup_tab"]])

    with login_tab:
        with st.form("login_form"):
            login_username = st.text_input(TEXT_RO["username_label"], key="login_uname")
            login_password = st.text_input(
                TEXT_RO["password_label"], type="password", key="login_pass"
            )
            login_button = st.form_submit_button(TEXT_RO["login_button"])

            if login_button:
                if check_user_db(login_username, login_password):
                    st.session_state.logged_in = True
                    st.session_state.username = login_username
                    st.session_state.current_pil_image = None
                    st.session_state.current_image_bytes_for_display = None
                    st.session_state.current_predictions = None
                    st.session_state.current_filename = None
                    st.session_state.current_image_id_in_db = None
                    st.session_state.feedback_submitted_for_current_image = False
                    st.success(TEXT_RO["login_success"])
                    st.rerun()
                else:
                    st.error(TEXT_RO["login_failed"])

    with signup_tab:
        with st.form("signup_form"):
            signup_username = st.text_input(
                TEXT_RO["username_label"], key="signup_uname"
            )
            signup_password = st.text_input(
                TEXT_RO["password_label"], type="password", key="signup_pass"
            )
            signup_button = st.form_submit_button(TEXT_RO["signup_button"])

            if signup_button:
                success, message = add_user_db(signup_username, signup_password)
                if success:
                    st.success(TEXT_RO["signup_success"])
                else:
                    if (
                        "goale" in message
                    ):  # "Numele de utilizator și parola nu pot fi goale."
                        st.error(TEXT_RO["signup_failed_format"])
                    elif (
                        "există deja" in message
                    ):  # "Numele de utilizator există deja."
                        st.error(TEXT_RO["signup_failed_exists"])
                    else:
                        st.error(TEXT_RO["signup_failed_general"].format(e=message))


# --- Main Application Flow ---
if not st.session_state.logged_in:
    show_login_register_forms()
else:
    try:
        sidebar_image_path = Path("sidebar_image.jpg")
        if sidebar_image_path.exists():
            sidebar_image = Image.open(sidebar_image_path)
            sidebar_image.thumbnail((1024, 1024))

            st.sidebar.image(sidebar_image, caption="Exemplu Stil Neoromânesc")
        else:
            st.sidebar.image(
                "https://https://upload.wikimedia.org/wikipedia/commons/e/e7/Castelul_Cantacuzino_02.jpg",
                caption="Exemplu Stil Neoromânesc\n (Castelul Cantacuzino, Bușteni)",
            )
    except Exception:  # În cazul în care eșuează imaginea locală
        st.sidebar.image(
            "https://https://upload.wikimedia.org/wikipedia/commons/e/e7/Castelul_Cantacuzino_02.jpg",
            caption="Exemplu Stil Neoromânesc\n (Castelul Cantacuzino, Bușteni)",
        )

    st.sidebar.header(TEXT_RO["sidebar_navigation_header"])

    available_models_list = list_models(MODELS_DIR)
    if not available_models_list:
        st.sidebar.error(f"Niciun model .pkl găsit în directorul '{MODELS_DIR}'.")
        st.session_state.learner = None
    else:
        default_model_file_sb = (
            DEFAULT_MODEL_NAME
            if DEFAULT_MODEL_NAME in available_models_list
            else available_models_list[0]
        )
        current_selected_model_name_sb = (
            Path(st.session_state.current_model_path).name
            if st.session_state.current_model_path
            and Path(st.session_state.current_model_path).name in available_models_list
            else default_model_file_sb
        )

        try:
            idx_sb = available_models_list.index(current_selected_model_name_sb)
        except ValueError:
            idx_sb = (
                available_models_list.index(default_model_file_sb)
                if default_model_file_sb in available_models_list
                else 0
            )
            st.session_state.current_model_path = str(
                MODELS_DIR / available_models_list[idx_sb]
            )

        selected_model_name_sb = st.sidebar.selectbox(
            "Selectați Modelul Antrenat (.pkl):",
            options=available_models_list,
            index=idx_sb,
            key="model_selector_app",
        )

        new_model_path_sb = str(MODELS_DIR / selected_model_name_sb)
        if (
            new_model_path_sb != st.session_state.current_model_path
            or st.session_state.learner is None
        ):
            with st.spinner(f"Se încarcă modelul {selected_model_name_sb}..."):
                st.session_state.current_model_path = new_model_path_sb
                st.session_state.learner, KNOWN_ARCHITECTURAL_STYLES = load_learner_and_vocab(
                    st.session_state.current_model_path
                )
                if st.session_state.learner is None:
                    st.sidebar.error(
                        f"Nu s-a putut încărca modelul {selected_model_name_sb}."
                    )

    # Trimite st.session_state direct către pagini; acestea vor accesa st.session_state.username
    page_options_map = {
        TEXT_RO["analyze_button"]: lambda: main_app_page(
            st.session_state,
            st.session_state.learner,
            TEXT_RO,
            list(KNOWN_ARCHITECTURAL_STYLES),
        ),
        TEXT_RO["gallery_title"]: lambda: gallery_page(st.session_state, TEXT_RO),
        TEXT_RO["map_title"]: lambda: map_favorites_page(st.session_state, TEXT_RO),
        TEXT_RO["about_title"]: lambda: thesis_info_page(TEXT_RO),
    }

    selected_page_title_nav = st.sidebar.radio(
        TEXT_RO["sidebar_goto_prompt"],
        options=list(page_options_map.keys()),
        key="navigation_radio_app",
    )

    st.sidebar.markdown("---")
    st.sidebar.info(
        f"""
        **{TEXT_RO["sidebar_info_title"]}**
        {TEXT_RO["sidebar_info_focus"]}

        {TEXT_RO["sidebar_info_upload_instruction"]}
        """
    )
    st.sidebar.markdown(f"**{TEXT_RO['logged_in_as']}** `{st.session_state.username}`")

    if st.sidebar.button(TEXT_RO["logout_button"], key="logout_btn_app"):
        st.session_state.logged_in = False
        st.session_state.username = None
        st.session_state.current_pil_image = None
        st.session_state.current_image_bytes_for_display = None
        st.session_state.current_predictions = None
        st.session_state.current_filename = None
        st.session_state.current_image_id_in_db = None
        st.session_state.feedback_submitted_for_current_image = False
        st.rerun()

    if selected_page_title_nav in page_options_map:
        page_options_map[selected_page_title_nav]()
    else:
        list(page_options_map.values())[0]()

# Footer
st.markdown("---")
st.markdown(
    f"<div style='text-align: center; color: gray;'>{TEXT_RO['footer_text']}</div>",
    unsafe_allow_html=True,
)

if __name__ == "__main__":
    if platform.system() == "Windows":
        multiprocessing.freeze_support()
