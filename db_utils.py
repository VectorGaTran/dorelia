import sqlite3
from pathlib import Path
import json
import time
from PIL import Image
import io
import streamlit as st
from werkzeug.security import (
    generate_password_hash,
    check_password_hash,
)  # Importă funcțiile corecte

DB_PATH = Path("user_data.db")
USER_IMAGE_DIR = Path("user_uploaded_images")
USER_IMAGE_DIR.mkdir(parents=True, exist_ok=True)

TEXT_RO_DB = {
    "style_unidentified": "Neidentificat",
    "style_mix": "Amestec {style1} + {style2}",
}


def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL
        )
    """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS gallery (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            filename TEXT NOT NULL,
            image_path TEXT NOT NULL UNIQUE,
            primary_style TEXT,
            primary_confidence REAL,
            predictions_json TEXT,
            notes TEXT,
            address TEXT
        )
    """
    )
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS map_locations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            address TEXT NOT NULL,
            latitude REAL NOT NULL,
            longitude REAL NOT NULL,
            style TEXT,
            image_preview_path TEXT,
            filename TEXT
        )
    """
    )
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            image_db_id INTEGER,
            image_filename TEXT,
            predicted_style TEXT,
            user_corrected_style TEXT,
            is_correct INTEGER,
            comment TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """
    )
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS style_details (
            style_name TEXT PRIMARY KEY,
            period_info TEXT,
            region_info TEXT,
            more_info_text TEXT,
            more_info_link TEXT
        )
    """
    )
    conn.commit()
    conn.close()


# def _hash_password(password):
#     """un hash de bază, nerecomandat pentru producție"""
#     return hashlib.sha256(password.encode('utf-8')).hexdigest()


def add_user_db(username, password):
    if not username or not password:
        return False, "Numele de utilizator și parola nu pot fi goale."
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        # Folosește generate_password_hash pentru a crea un hash sigur (cu salt)
        hashed_password = generate_password_hash(password)
        cursor.execute(
            "INSERT INTO users (username, password_hash) VALUES (?, ?)",
            (username, hashed_password),
        )
        conn.commit()
        return True, "Utilizator înregistrat cu succes."
    except sqlite3.IntegrityError:
        return False, "Numele de utilizator există deja."
    except Exception as e:
        return False, f"Eroare la înregistrare: {e}"
    finally:
        conn.close()


def check_user_db(username, password):
    """verifică dacă numele utilizatorului și parola se potrivesc"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT password_hash FROM users WHERE username = ?", (username,))
    result = cursor.fetchone()
    conn.close()
    if result:
        # Folosește check_password_hash care compară parola dată cu hash-ul stocat
        if check_password_hash(result[0], password):
            return True
    return False


def save_image(image_bytes, original_filename, username, is_thumbnail=False):
    user_dir = USER_IMAGE_DIR / username
    if is_thumbnail:
        user_dir = user_dir / "thumbnails"
    user_dir.mkdir(parents=True, exist_ok=True)

    file_extension = Path(original_filename).suffix or ".png"
    safe_original_filename_stem = "".join(
        c if c.isalnum() else "_" for c in Path(original_filename).stem
    )
    new_filename = f"{safe_original_filename_stem}{file_extension}"
    image_path = user_dir / new_filename
    try:
        with open(image_path, "wb") as f:
            f.write(image_bytes)
        return str(image_path)
    except Exception as e:
        st.error(f"Eroare la salvarea imaginii pe disc: {e}")
        return None


def add_to_gallery_db(
    username, filename, image_path, predictions_list, notes, address, TEXT_RO_SESSION
):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    primary_style = TEXT_RO_SESSION.get("style_unidentified", "Neidentificat")
    primary_confidence = 0.0

    if predictions_list and len(predictions_list) > 0:
        # Determină stilul principal după încredere peste 40% sau amestec de stiluri
        high_conf_preds = [p for p in predictions_list if p["confidence"] >= 0.40]
        high_conf_preds.sort(key=lambda x: x["confidence"], reverse=True)

        if not high_conf_preds:
            primary_style = TEXT_RO_SESSION.get("style_unidentified", "Neidentificat")
            primary_confidence = (
                predictions_list[0]["confidence"] if predictions_list else 0.0
            )
        elif len(high_conf_preds) == 1:
            primary_style = high_conf_preds[0]["style"]
            primary_confidence = high_conf_preds[0]["confidence"]
        else:  # len >= 2
            style1 = high_conf_preds[0]["style"]
            style2 = high_conf_preds[1]["style"]
            primary_style = TEXT_RO_SESSION.get(
                "style_mix", "Amestec {style1} + {style2}"
            ).format(style1=style1, style2=style2)
            primary_confidence = (
                high_conf_preds[0]["confidence"] + high_conf_preds[1]["confidence"]
            ) / 2

    predictions_json = json.dumps(predictions_list)
    try:
        cursor.execute(
            """
            INSERT INTO gallery (username, filename, image_path, primary_style, primary_confidence, predictions_json, notes, address)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                username,
                filename,
                image_path,
                primary_style,
                primary_confidence,
                predictions_json,
                notes,
                address,
            ),
        )
        conn.commit()
        return cursor.lastrowid
    except sqlite3.IntegrityError:
        # În cazul în care imaginea există deja (deși ar trebui să fie unică) se caută intrarea existentă pentru returnarea ID-ului
        cursor.execute(
            "SELECT id FROM gallery WHERE image_path = ? AND username = ?",
            (image_path, username),
        )  # verifică și pentru utilizator
        existing = cursor.fetchone()
        if existing:
            return existing[0]
        return None
    except Exception as e:
        st.error(f"Eroare la adăugarea în galerie (DB): {e}")
        return None
    finally:
        conn.close()


def get_gallery_from_db(username, text_ro_config=None):
    current_text_ro = text_ro_config if text_ro_config else TEXT_RO_DB
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT id, filename, image_path, primary_style, primary_confidence, predictions_json, notes, address
        FROM gallery
        WHERE username = ? ORDER BY id DESC
    """,
        (username,),
    )
    items = cursor.fetchall()
    conn.close()

    gallery_data = []
    for item in items:
        image_bytes_data = None
        try:
            if Path(item[2]).exists():
                with open(item[2], "rb") as f:
                    image_bytes_data = f.read()
            else:
                print(f"Atenție: Calea imaginii nu există pentru galerie: {item[2]}")
        except Exception as e:
            print(f"Eroare la citirea imaginii din galerie: {item[2]}, {e}")

        predictions_list = json.loads(item[5]) if item[5] else []

        gallery_data.append(
            {
                "id": item[0],
                "filename": item[1],
                "image_path": item[2],
                "primary_style": (
                    item[3]
                    if item[3]
                    else current_text_ro.get("style_unidentified", "Neidentificat")
                ),
                "primary_confidence": item[4] if item[4] is not None else 0.0,
                "predictions": predictions_list,
                "notes": item[6],
                "address": item[7],
                "image_bytes": image_bytes_data,
            }
        )
    return gallery_data


def remove_from_gallery_db(item_id, username):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute(
            "SELECT image_path FROM gallery WHERE id = ? AND username = ?",
            (item_id, username),
        )
        result = cursor.fetchone()
        if result:
            image_path_to_delete = Path(result[0])
            cursor.execute(
                "DELETE FROM gallery WHERE id = ? AND username = ?", (item_id, username)
            )
            conn.commit()
            if conn.total_changes > 0:
                cursor.execute(
                    "SELECT COUNT(*) FROM gallery WHERE image_path = ? AND username = ?",
                    (str(image_path_to_delete), username),
                )
                gallery_references = cursor.fetchone()[0]
                cursor.execute(
                    "SELECT COUNT(*) FROM map_locations WHERE image_preview_path = ? AND username = ?",
                    (str(image_path_to_delete), username),
                )
                map_references = cursor.fetchone()[0]

                if (
                    image_path_to_delete.exists()
                    and gallery_references == 0
                    and map_references == 0
                ):
                    try:
                        image_path_to_delete.unlink()
                    except Exception as e_file_delete:
                        st.warning(
                            f"Imaginea a fost scoasă din baza de date, dar a eșuat ștergerea fișierului: {e_file_delete}"
                        )
                return True
            return False
        return False
    except Exception as e:
        st.error(f"Eroare la ștergerea din galerie: {e}")
        return False
    finally:
        conn.close()


def add_to_map_db(
    username, address, latitude, longitude, style, image_preview_path, filename
):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            INSERT INTO map_locations (username, address, latitude, longitude, style, image_preview_path, filename)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
            (
                username,
                address,
                latitude,
                longitude,
                style,
                image_preview_path,
                filename,
            ),
        )
        conn.commit()
        return cursor.lastrowid
    except Exception as e:
        st.error(f"Eroare la adăugarea pe hartă: {e}")
        return None
    finally:
        conn.close()


def get_map_locations_from_db(username, text_ro_config=None):
    current_text_ro = text_ro_config if text_ro_config else TEXT_RO_DB
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT id, address, latitude, longitude, style, image_preview_path, filename
        FROM map_locations
        WHERE username = ?
    """,
        (username,),
    )
    items = cursor.fetchall()
    conn.close()

    locations_data = []
    for item in items:
        preview_bytes = None
        if item[5] and Path(item[5]).exists():
            try:
                with open(item[5], "rb") as f:
                    preview_bytes = f.read()
            except Exception as e_img:
                print(
                    f"Eroare la citirea thumbnail-ului pentru hartă: {item[5]}, {e_img}"
                )

        locations_data.append(
            {
                "id": item[0],
                "address": item[1],
                "latitude": item[2],
                "longitude": item[3],
                "style": (
                    item[4]
                    if item[4]
                    else current_text_ro.get("style_unidentified", "Neidentificat")
                ),
                "image_preview_path": item[5],
                "filename": item[6],
                "image_preview_bytes": preview_bytes,
            }
        )
    return locations_data


def remove_from_map_db(location_ids, username):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    removed_count = 0
    try:
        for loc_id in location_ids:
            cursor.execute(
                "SELECT image_preview_path FROM map_locations WHERE id = ? AND username = ?",
                (loc_id, username),
            )
            result = cursor.fetchone()
            thumb_path_to_delete = None
            if result and result[0]:
                thumb_path_to_delete = Path(result[0])

            cursor.execute(
                "DELETE FROM map_locations WHERE id = ? AND username = ?",
                (loc_id, username),
            )
            if cursor.rowcount > 0:
                removed_count += 1
                if thumb_path_to_delete and thumb_path_to_delete.exists():
                    cursor.execute(
                        "SELECT COUNT(*) FROM map_locations WHERE image_preview_path = ? AND username = ?",
                        (str(thumb_path_to_delete), username),
                    )
                    map_references = cursor.fetchone()[0]
                    if map_references == 0:
                        try:
                            thumb_path_to_delete.unlink()
                        except Exception as e_file_del_map:
                            st.warning(
                                f"Locația a fost scoasă din baza de date, dar ștergerea thumbnail-ului a eșuat: {e_file_del_map}"
                            )
        conn.commit()
    except Exception as e:
        st.error(f"Eroare la ștergerea de pe hartă: {e}")
    finally:
        conn.close()
    return removed_count


def add_feedback_db(
    username,
    image_db_id,
    image_filename,
    predicted_style,
    is_correct,
    user_corrected_style=None,
    comment=None,
):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            INSERT INTO feedback (username, image_db_id, image_filename, predicted_style, is_correct, user_corrected_style, comment)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
            (
                username,
                image_db_id,
                image_filename,
                predicted_style,
                1 if is_correct else 0,
                user_corrected_style,
                comment,
            ),
        )
        conn.commit()
    except Exception as e:
        st.error(f"Eroare la salvarea feedback-ului: {e}")
    finally:
        conn.close()


def get_style_details(style_name):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT period_info, region_info, more_info_text, more_info_link FROM style_details WHERE style_name = ?",
        (style_name,),
    )
    details = cursor.fetchone()
    conn.close()
    if details:
        return {
            "period": details[0],
            "region": details[1],
            "details_text": details[2],
            "details_link": details[3],
        }
    return None


def add_or_update_style_details(
    style_name, period_info, region_info, more_info_text, more_info_link
):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT OR REPLACE INTO style_details (style_name, period_info, region_info, more_info_text, more_info_link)
        VALUES (?, ?, ?, ?, ?)
    """,
        (style_name, period_info, region_info, more_info_text, more_info_link),
    )
    conn.commit()
    conn.close()


init_db()  # Inițializează baza de date la importul modulului

if __name__ == "__main__":
    print("Se inițializează baza de date și se populează detaliile despre stiluri...")
    add_or_update_style_details(
        style_name="Achaemenid architecture",
        period_info="Aproximativ 550 î.Hr. - 330 î.Hr.",
        region_info="Imperiul Persan (Iranul de astăzi și zonele învecinate)",
        more_info_text="""Arhitectura ahemenidă, specifică Imperiului Persan, este cunoscută pentru construcții monumentale menite să glorifice imperiul. Printre clădirile emblematice, caracterizate prin utilizarea de coloane înalte și reliefuri sculptate elaborate, includ:
    \n- Palatul Apadana de la Persepolis, Iran (finalizat în prima jumătate a sec. V î.Hr.)
    \n- Palatul lui Darius de la Susa, Iran (construit în cca. 515-505 î.Hr.)""",
        more_info_link="https://en.wikipedia.org/wiki/Achaemenid_architecture",
    )
    add_or_update_style_details(
        style_name="American Foursquare architecture",
        period_info="Mijlocul anilor 1890 - Sfârșitul anilor 1930",
        region_info="Statele Unite ale Americii",
        more_info_text="""Stilul American Foursquare, o reacție la stilurile ornamentate victoriene, se caracterizează printr-o formă cubică. Exemple tipice sunt abundente în suburbiile americane vechi:
    \n- Casa B. Harley Bradley, Kankakee, Illinois (proiectată de Frank Lloyd Wright, 1900)
    \n- Casele din cataloagele Sears, Roebuck and Co. (populare în anii 1910-1930)""",
        more_info_link="https://en.wikipedia.org/wiki/American_Foursquare",
    )
    add_or_update_style_details(
        style_name="American craftsman style",
        period_info="Sfârșitul sec. XIX - Începutul sec. XX",
        region_info="Statele Unite ale Americii",
        more_info_text="""Stilul American Craftsman promovează manopera de calitate și materialele naturale. Casele au adesea acoperișuri joase cu streșini largi. Exemple faimoase includ:
    \n- Casa Gamble (Gamble House), Pasadena, California (construită în 1908)
    \n- Casa Roycroft, East Aurora, New York (extinsă în stil Craftsman la începutul sec. XX)""",
        more_info_link="https://en.wikipedia.org/wiki/American_Craftsman",
    )
    add_or_update_style_details(
        style_name="Ancient Egyptian architecture",
        period_info="Aproximativ 3100 î.Hr. - 30 î.Hr.",
        region_info="Egiptul Antic",
        more_info_text="""Arhitectura egipteană antică este renumită pentru structurile sale monumentale construite din piatră. Exemple iconice, care au rezistat milenii, sunt:
    \n- Marile Piramide din Giza (finalizate în cca. 2560 î.Hr.)
    \n- Complexul de temple de la Karnak (dezvoltat între cca. 2000–30 î.Hr.)""",
        more_info_link="https://ro.wikipedia.org/wiki/Arhitectura_egiptean%C4%83_antic%C4%83",
    )
    add_or_update_style_details(
        style_name="Andean Baroque Architecture",
        period_info="Aproximativ 1680 - 1780",
        region_info="Regiunea Anzilor din America de Sud (Peru, Bolivia, Ecuador)",
        more_info_text="""Barocul andin este o fuziune a stilului baroc european cu elemente indigene, remarcându-se prin ornamente bogate. Exemple notabile sunt:
    \n- Biserica Companiei de Isus (Iglesia de la Compañía de Jesús), Cusco, Peru (finalizată în 1668)
    \n- Biserica San Lorenzo de Carangas, Potosí, Bolivia (fațada finalizată în 1744)""",
        more_info_link="https://en.wikipedia.org/wiki/Andean_Baroque",
    )
    add_or_update_style_details(
        style_name="Art Deco architecture",
        period_info="Anii 1920 - 1940",
        region_info="Răspândire globală, inclusiv România (mai ales București)",
        more_info_text="""Art Deco este un stil caracterizat prin forme geometrice bogate, ornamentație luxoasă și utilizarea de materiale moderne precum oțelul și aluminiul. Este un stil al eleganței și modernității, și reprezenta la vremea lui o oarecare încredere în progresul social și cel tehnologic. Clădiri emblematice includ:
    \n- Palatul Telefoanelor, București, România (inaugurat în 1934)
    \n- Chrysler Building, New York, SUA (finalizat în 1930)""",
        more_info_link="https://ro.wikipedia.org/wiki/Art_Deco",
    )
    add_or_update_style_details(
        style_name="Art Nouveau architecture",
        period_info="Aproximativ 1890 - 1910",
        region_info="Răspândire globală, cu centre importante în Franța, Belgia, Spania, Austria, Ungaria și România",
        more_info_text="""Arhitectura Art Nouveau se inspiră din formele organice ale naturii, utilizând linii curbe și decorațiuni florale. Exemple reprezentative sunt:
    \n- Palatul Vulturul Negru, Oradea, România (construit în 1907-1908)
    \n- Casa Batlló, Barcelona, Spania (remodelată de Gaudí între 1904-1906)""",
        more_info_link="https://ro.wikipedia.org/wiki/Art_Nouveau",
    )
    add_or_update_style_details(
        style_name="Baroque architecture",
        period_info="Începutul sec. XVII - Mijlocul sec. XVIII",
        region_info="Europa, cu origini în Italia",
        more_info_text="""Arhitectura barocă este un stil grandios și dramatic, caracterizat prin opulență, forme curbe, detalii elaborate și jocuri de lumină și umbră. A fost un stil al Contrareformei, menit să impresioneze și să emoționeze. Exemple notabile sunt:
    \n- Palatul Brukenthal, Sibiu, România (construit în 1778-1788)
    \n- Palatul de la Versailles, Franța (extins masiv între 1661-1715)""",
        more_info_link="https://ro.wikipedia.org/wiki/Arhitectur%C4%83_baroc%C4%83",
    )
    add_or_update_style_details(
        style_name="Bauhaus architecture",
        period_info="Anii 1919 - 1933",
        region_info="Germania, cu influențe globale ulterioare",
        more_info_text="""Școala Bauhaus a promovat o arhitectură funcționalistă, rațională, modernă și minimalistă, cu o influență majoră asupra stilului internațional. Clădirile emblematice ale stilului sunt:
    \n- Clădirea Bauhaus, Dessau, Germania (proiectată de Walter Gropius, 1925-1926)
    \n- Complexul de locuințe Weissenhof, Stuttgart, Germania (construit în 1927)""",
        more_info_link="https://ro.wikipedia.org/wiki/Bauhaus",
    )
    add_or_update_style_details(
        style_name="Beaux-Arts architecture",
        period_info='Aproximativ 1830 - 1940 (România: 1878-1916, "la belle epoque")',
        region_info="Global cu exemple notabile în Franța și Statele Unite ale Americii, dar și în România",
        more_info_text="""Stilul Beaux-Arts este un stil academic, neoclasic, caracterizat prin simetrie, grandoare, decorațiuni sculpturale elaborate și utilizarea de coloane, frontoane și alte elemente clasice. A fost popular pentru clădiri publice și monumentale. În București, exemple remarcabile sunt:
    \n- Palatul CEC, București, România (finalizat în 1900)
    \n- Palatul Cantacuzino (Muzeul Național 'George Enescu'), București, România (inaugurat în 1903)""",
        more_info_link="https://en.wikipedia.org/wiki/Beaux-Arts_architecture",
    )
    add_or_update_style_details(
        style_name="Blobitecture",
        period_info="Mijlocul anilor 1990 - Prezent",
        region_info="Global",
        more_info_text="""Blobitecture,, utilizează forme organice și fluide. Exemple faimoase care definesc acest curent sunt:
    \n- Kunsthaus Graz, Graz, Austria (deschis în 2003)
    \n- Centrul de muzică Sage Gateshead, Gateshead, Anglia (deschis în 2004)""",
        more_info_link="https://en.wikipedia.org/wiki/Blobitecture",
    )
    add_or_update_style_details(
        style_name="Brutalism",
        period_info="Anii 1950 - Anii 1970",
        region_info="Răspândire globală, inclusiv România",
        more_info_text="""Brutalismul este un stil arhitectural modern caracterizat prin utilizarea betonului aparent (béton brut), forme masive, geometrice și o estetică nefinisată. A fost adesea asociat cu proiecte de locuințe sociale și clădiri guvernamentale. Exemple notabile includ:
    \n- Palatul Administrativ Ploiești, București, România (inaugurat în 1971)
    \n- Habitat 67, Montreal, Canada (construit în 1967)""",
        more_info_link="https://ro.wikipedia.org/wiki/Brutalism",
    )
    add_or_update_style_details(
        style_name="Byzantine architecture",
        period_info="330 d.Hr. - 1453 d.Hr.",
        region_info="Imperiul Bizantin (Europa de Est și Orientul Mijlociu)",
        more_info_text="""Arhitectura bizantină a dezvoltat caracteristici proprii, precum domurile și mozaicurile interioare bogate, influențând profund arhitectura religioasă ulterioară din România. Exemple iconice ale stilului sunt:
    \n- Hagia Sophia, Istanbul, Turcia (finalizată în 537 d.Hr.)
    \n- Bazilica San Vitale, Ravenna, Italia (finalizată în 547 d.Hr.)""",
        more_info_link="https://ro.wikipedia.org/wiki/Arhitectura_bizantin%C4%83",
    )
    add_or_update_style_details(
        style_name="Chicago school architecture",
        period_info="Sfârșitul sec. XIX - Începutul sec. XX",
        region_info="Chicago, Statele Unite ale Americii",
        more_info_text="""Școala de la Chicago a fost un grup de arhitecți pionieri în construcția de zgârie-nori cu structură metalică. Stilul lor a pus accent pe funcționalitate, ferestre mari (ferestre Chicago) și o ornamentație limitată. Clădiri de referință includ:
    \n- Carson, Pirie, Scott and Company Building, Chicago, SUA (construit în 1899)
    \n- Auditorium Building, Chicago, SUA (finalizat în 1889)""",
        more_info_link="https://en.wikipedia.org/wiki/Chicago_school_(architecture)",
    )
    add_or_update_style_details(
        style_name="Colonial Revival archtecture",
        period_info="Sfârșitul sec. XIX - Mijlocul sec. XX",
        region_info="Statele Unite ale Americii",
        more_info_text="""Stilul Colonial Revival reinterpretează elemente din arhitectura colonială americană. Exemple notabile pot fi văzute în:
    \n- Clădirile reconstruite din Colonial Williamsburg, Virginia (reconstrucție începută în anii 1920)
    \n- Numeroase case suburbane din SUA, precum cele din Shaker Heights, Ohio (dezvoltat în anii 1910-1940)""",
        more_info_link="https://en.wikipedia.org/wiki/Colonial_Revival_architecture",
    )
    add_or_update_style_details(
        style_name="Colonial architecture",
        period_info="Sec. XVII - Sec. XIX",
        region_info="Colonii europene din America, Africa și Asia",
        more_info_text="""Arhitectura colonială adaptează stilurile europene la noile medii. Exemple reprezentative pentru diferitele puteri coloniale includ:
    - Palatul Național, Mexico City, Mexic (Stil Colonial Spaniol, construit începând cu 1522 pe ruinele palatului lui Moctezuma)
    - Independence Hall, Philadelphia, SUA (Stil Colonial Britanic/Georgian, finalizat în 1753)""",
        more_info_link="https://en.wikipedia.org/wiki/Colonial_architecture",
    )
    add_or_update_style_details(
        style_name="Deconstructivism",
        period_info="Sfârșitul anilor 1980 - Prezent",
        region_info="Global",
        more_info_text="""Deconstructivismul se opune raționalității ordonate, caracterizându-se prin fragmentare și forme non-rectilinii. Exemple care au definit mișcarea sunt:
    \n- Muzeul Guggenheim, Bilbao, Spania (proiectat de Frank Gehry, 1997)
    \n- Vitra Design Museum, Weil am Rhein, Germania (proiectat de Frank Gehry, 1989)""",
        more_info_link="https://ro.wikipedia.org/wiki/Deconstructivism",
    )
    add_or_update_style_details(
        style_name="Earthquake Baroque Architecture",
        period_info="Sec. XVII - Sec. XVIII",
        region_info="Filipine și Guatemala",
        more_info_text="""Barocul 'seismic' este o adaptare a stilului baroc la condițiile seismice, folosind contraforturi masive și ziduri groase. Exemple remarcabile sunt:
    \n- Biserica San Agustin, Paoay, Filipine (finalizată în 1710)
    \n- Biserica La Merced, Antigua, Guatemala (finalizată în 1767)""",
        more_info_link="https://en.wikipedia.org/wiki/Earthquake_Baroque",
    )
    add_or_update_style_details(
        style_name="Eco-architecture",
        period_info="Sfârșitul sec. XX - Prezent",
        region_info="Global",
        more_info_text="""Arhitectura ecologică urmărește minimizarea impactului asupra mediului. Exemple iconice la nivel mondial sunt:
    \n- Complexul Eden Project, Cornwall, Marea Britanie (deschis în 2001)
    \n- Turnurile rezidențiale Bosco Verticale, Milano, Italia (inaugurate în 2014)""",
        more_info_link="https://en.wikipedia.org/wiki/Sustainable_architecture",
    )
    add_or_update_style_details(
        style_name="Edwardian architecture",
        period_info="1901 - 1914",
        region_info="Imperiul Britanic",
        more_info_text="""Arhitectura edwardiană este mai puțin ornamentată decât cea victoriană, fiind influențată de mișcarea Arts and Crafts și de stilul baroc continental (Beaux-Arts). Exemple de seamă includ:
    \n- Hotelul Ritz, Londra, Marea Britanie (deschis în 1906)
    \n- Port of Liverpool Building, Liverpool, Marea Britanie (construit în 1904-1907)""",
        more_info_link="https://en.wikipedia.org/wiki/Edwardian_architecture",
    )
    add_or_update_style_details(
        style_name="French Renaissance Architecture",
        period_info="Sfârșitul sec. XV - Începutul sec. XVII",
        region_info="Franța",
        more_info_text="""Arhitectura renascentistă franceză a combinat formele gotice târzii cu cele renascentiste italiene, fiind cunoscută pentru castelele de pe Valea Loarei. Clădiri emblematice sunt:
    \n- Castelul Chambord (Château de Chambord), Franța (construit între 1519-1547)
    \n- Aripa Lescot a Palatului Luvru (Palais du Louvre), Paris, Franța (construită între 1546-1551)""",
        more_info_link="https://en.wikipedia.org/wiki/French_Renaissance_architecture",
    )
    add_or_update_style_details(
        style_name="Georgian architecture",
        period_info="Aproximativ 1714 - 1830",
        region_info="Marea Britanie și coloniile sale",
        more_info_text="""Arhitectura georgiană este un stil elegant și simetric, inspirat de arhitectura clasică, caracterizat prin proporții armonioase. Exemple faimoase sunt:
    \n- The Circus, Bath, Marea Britanie (construit între 1754-1768)
    \n- Royal Crescent, Bath, Marea Britanie (construit între 1767-1774)""",
        more_info_link="https://en.wikipedia.org/wiki/Georgian_architecture",
    )
    add_or_update_style_details(
        style_name="Gothic architecture",
        period_info="Mijlocul sec. XII - Sec. XVI",
        region_info="Europa",
        more_info_text="""Arhitectura gotică a revoluționat construcția prin introducerea arcului frânt, a bolții pe ogive și a contraforților. Aceste inovații au permis construirea de catedrale înalte, cu ziduri subțiri și vitralii imense. Exemple majore din România și din lume sunt:
    \n- Biserica Neagră, Brașov, România (construită în sec. XIV-XV)
    \n- Castelul Corvinilor, Hunedoara, România (construit în sec. XV)
    \n- Catedrala Notre-Dame, Paris, Franța (construită între 1163-1345)""",
        more_info_link="https://ro.wikipedia.org/wiki/Arhitectur%C4%83_gotic%C4%83",
    )
    add_or_update_style_details(
        style_name="Greek Revival architecture",
        period_info="Sfârșitul sec. XVIII - Mijlocul sec. XIX",
        region_info="Europa și Statele Unite ale Americii",
        more_info_text="""Stilul neo-grec s-a inspirat direct din templele Greciei antice, fiind un substil al neoclasicismului. Se caracterizează prin utilizarea ordinelor clasice (doric, ionic, corintic), frontoane și colonade. Exemple internaționale de prim rang sunt:
    \n- British Museum, Londra, Marea Britanie (fațada sudică finalizată în 1842)
    \n- Altes Museum, Berlin, Germania (construit între 1823-1830)""",
        more_info_link="https://en.wikipedia.org/wiki/Greek_Revival_architecture",
    )
    add_or_update_style_details(
        style_name="Herodian architecture",
        period_info="37 î.Hr. - 4 î.Hr.",
        region_info="Iudeea (Israelul și Palestina de astăzi)",
        more_info_text="""Arhitectura herodiană, denumită după regele Irod cel Mare, a fost un stil monumental care a combinat tradițiile locale cu tehnicile de construcție romane. Cele mai faimoase proiecte ale lui Irod cel Mare au fost:
    \n- Reconstrucția celui de-al Doilea Templu din Ierusalim (începută în 20-19 î.Hr.), al cărui Zid al Plângerii este un vestigiu
    \n- Fortăreața montană și complexul de palate de la Masada (construită între 37-31 î.Hr.)""",
        more_info_link="https://en.wikipedia.org/wiki/Herodian_architecture",
    )
    add_or_update_style_details(
        style_name="International style",
        period_info="Anii 1920 - Anii 1970",
        region_info="Global",
        more_info_text="""Stilul internațional este o ramură majoră a arhitecturii moderne, caracterizată prin forme rectilinii, suprafețe netede, lipsa ornamentației și utilizarea sticlei, oțelului și betonului. A promovat o estetică universală, independentă de contextul cultural. Clădiri iconice la nivel global sunt:
    \n- Seagram Building, New York, SUA (proiectat de Ludwig Mies van der Rohe, 1958)
    \n- Villa Savoye, Poissy, Franța (proiectată de Le Corbusier, 1928-1931)""",
        more_info_link="https://en.wikipedia.org/wiki/International_Style_(architecture)",
    )
    add_or_update_style_details(
        style_name="Mannerist Architecture",
        period_info="Aproximativ 1520 - 1600",
        region_info="Italia, răspândindu-se apoi în Europa",
        more_info_text="""Arhitectura manieristă a fost o reacție la armonia clasică a Renașterii. Se caracterizează prin complexitate, ambiguitate spațială și utilizarea neașteptată a elementelor clasice, adesea pentru a crea surpriză și tensiune.Exemple cheie includ:
    \n- Palazzo Te, Mantova, Italia (proiectat de Giulio Romano, 1524-1534)
    \n- Vestibulul Bibliotecii Laurențiane, Florența, Italia (proiectat de Michelangelo, 1524)""",
        more_info_link="https://ro.wikipedia.org/wiki/Manierism#Arhitectura_manierist%C4%83",
    )
    add_or_update_style_details(
        style_name="Medieval Architecture",
        period_info="Aproximativ sec. V - sec. XV",
        region_info="Europa",
        more_info_text="""Arhitectura medievală a fost dominată de construcția de biserici, mănăstiri și castele, reflectând structura socială și religioasă a epocii.. Cuprinde stiluri majore precum Romanic și Gotic. Exemple reprezentative din România includ:
    \n- Catedrala Romano-Catolică Sfântul Mihail (aripa vestică romanică), Alba Iulia (sec. XIII)
    \n- Biserica Neagră (gotic), Brașov (sec. XIV-XV)""",
        more_info_link="https://en.wikipedia.org/wiki/Medieval_architecture",
    )
    add_or_update_style_details(
        style_name="Neo-futurism architecture",
        period_info="Sfârșitul sec. XX - Prezent",
        region_info="Global",
        more_info_text="""Neo-futurismul este un curent contemporan care se inspiră din estetica high-tech și din viziunile futuriste. Clădirile neo-futuriste explorează forme dinamice, tehnologii avansate și idei despre viitorul orașelor și al societății. Clădiri emblematice sunt:
    \n- The Shard, Londra, Marea Britanie (proiectat de Renzo Piano, 2012)
    \n- Orașul Artelor și Științelor (Ciudad de las Artes y las Ciencias), Valencia, Spania (proiectat de Santiago Calatrava și Félix Candela, 1998-2005)""",
        more_info_link="https://en.wikipedia.org/wiki/Neo-futurism",
    )
    add_or_update_style_details(
        style_name="Norman Architecture",
        period_info="Sec. XI - Sec. XII",
        region_info="Normandia, Anglia, sudul Italiei și Sicilia",
        more_info_text="""Arhitectura normandă este o ramură a stilului romanic, caracterizată prin masivitate, arce semicirculare și decorațiuni geometrice simple. Este cunoscută pentru castelele și catedralele sale impunătoare:
    \n- Turnul Alb (White Tower) din Turnul Londrei, Marea Britanie (construit în cca. 1078)
    \n- Catedrala din Durham, Marea Britanie (construită între 1093-1133)""",
        more_info_link="https://en.wikipedia.org/wiki/Norman_architecture",
    )
    add_or_update_style_details(
        style_name="Northern Renaissance Architecture",
        period_info="Sfârșitul sec. XV - Sec. XVI",
        region_info="Europa la nord de Alpi (Țările de Jos, Germania, Anglia, Franța)",
        more_info_text="""Renașterea nordică a adaptat idealurile renascentiste italiene la tradițiile locale, adesea păstrând elemente gotice. Se remarcă prin utilizarea cărămizii și a acoperișurilor înalte, cu frontoane în trepte. Exemple notabile:
    \n- Primăria din Anvers, Belgia (construită între 1561-1565)
    \n- Hardwick Hall, Derbyshire, Marea Britanie (finalizată în 1597)""",
        more_info_link="https://en.wikipedia.org/wiki/Northern_Renaissance#Architecture",
    )
    add_or_update_style_details(
        style_name="Orientalism Architecture",
        period_info="Sec. XIX",
        region_info="Europa și Statele Unite ale Americii",
        more_info_text="""Orientalismul în arhitectură a fost o modă romantică ce a imitat sau s-a inspirat din stilurile arhitecturale din Orientul Mijlociu, Africa de Nord și Asia. A folosit elemente precum arce în potcoavă, domuri și mozaicuri colorate. Exemple faimoase includ:
    \n- Royal Pavilion, Brighton, Marea Britanie (remodelat de John Nash, 1815-1823)
    \n- Marea Sinagogă din Florența, Italia (construită între 1874-1882)""",
        more_info_link="https://en.wikipedia.org/wiki/Orientalism#In_European_architecture_and_design",
    )
    add_or_update_style_details(
        style_name="Palladian architecture",
        period_info="Sec. XVI (Andrea Palladio) și revival în sec. XVII-XVIII",
        region_info="Italia (Veneto), cu un revival important în Marea Britanie și America de Nord",
        more_info_text="""Arhitectura palladiană se bazează pe principiile arhitectului italian Andrea Palladio, inspirate de templele romane antice. Se caracterizează prin simetrie strictă, proporții armonioase și utilizarea porticurilor cu fronton. Clădiri de referință sunt:
    \n- Villa Capra 'La Rotonda', Vicenza, Italia (proiectată de Palladio, 1567)
    \n- Casa Albă, Washington D.C., SUA (construită între 1792-1800, cu puternice influențe palladiene)""",
        more_info_link="https://en.wikipedia.org/wiki/Palladian_architecture",
    )
    add_or_update_style_details(
        style_name="Postmodern architecture",
        period_info="Anii 1960 - Anii 2000",
        region_info="Global",
        more_info_text="""Arhitectura postmodernă a fost o reacție împotriva stricteței stilului modern. A reintrodus ornamentul, culoarea, referințele istorice și umorul în designul clădirilor, adesea într-un mod eclectic și în unele cauri, oarecum ironic. Exemple includ:
    \n- 550 Madison Avenue (fostul AT&T Building), New York, SUA (proiectat de Philip Johnson, 1984)""",
        more_info_link="https://ro.wikipedia.org/wiki/Arhitectura_postmodern%C4%83",
    )
    add_or_update_style_details(
        style_name="Queen Anne architecture",
        period_info="Ultimul sfert al sec. XIX - Începutul sec. XX",
        region_info="Marea Britanie și Statele Unite ale Americii",
        more_info_text="""Stilul Queen Anne este pitoresc și eclectic, caracterizat prin asimetrie, turnuri, pridvoare, ferestre de diferite forme și o varietate de texturi pe fațadă.. Exemple reprezentative sunt:
    \n- Casa Carson (Carson Mansion), Eureka, California, SUA (construită în 1884-1886)
    \n- Diverse case din cartiere precum Bedford Park, Londra, Marea Britanie (dezvoltat în anii 1870)""",
        more_info_link="https://en.wikipedia.org/wiki/Queen_Anne_style_architecture",
    )
    add_or_update_style_details(
        style_name="Rococo Architecture",
        period_info="Anii 1730 - Anii 1760",
        region_info="Franța, răspândindu-se în Germania, Austria și alte părți ale Europei",
        more_info_text="""Rococo este o formă târzie și exuberantă a barocului, axată în special pe designul interior. Se caracterizează prin decorațiuni elaborate, forme asimetrice, motive de scoici și plante (rocaille) și culori pastelate. Palate faimoase pentru interioarele lor Rococo sunt:
    \n- Palatul Sanssouci, Potsdam, Germania (construit în 1745-1747)
    \n- Palatul Caterina, Pușkin, Rusia (extins și redecorat în anii 1750)""",
        more_info_link="https://en.wikipedia.org/wiki/Rococo_architecture",
    )
    add_or_update_style_details(
        style_name="Roman Classical architecture",
        period_info="Aproximativ 509 î.Hr. - Sec. IV d.Hr.",
        region_info="Imperiul Roman",
        more_info_text="""Arhitectura romană a preluat vocabularul clasic grecesc, dar l-a dezvoltat prin inovații precum arcul, bolta și domul, precum și prin utilizarea betonului. A excelat în construcții de inginerie civilă și clădiri publice la scară mare.. Exemple majore din imperiu, inclusiv de pe teritoriul României, sunt:
    \n- Tropaeum Traiani, Adamclisi, România (inaugurat în 109 d.Hr.)
    \n- Colosseumul, Roma, Italia (finalizat în 80 d.Hr.)
    \n- Panteonul, Roma, Italia (reconstruit în cca. 126 d.Hr.)""",
        more_info_link="https://ro.wikipedia.org/wiki/Arhitectur%C4%83_roman%C4%83",
    )
    add_or_update_style_details(
        style_name="Romanesque architecture",
        period_info="Sec. X - Sec. XII",
        region_info="Europa",
        more_info_text="""Arhitectura romanică este primul stil pan-european după cel roman. Se caracterizează prin ziduri groase, ferestre mici, arce semicirculare și o atmosferă generală de masivitate și soliditate. Este stilul castelelor și mănăstirilor medievale timpurii. Exemple notabile sunt:
    - Catedrala Romano-Catolică Sfântul Mihail (aripa vestică), Alba Iulia, România (sec. XIII)
    - Complexul Catedralei din Pisa (inclusiv Turnul înclinat), Italia (construit în sec. XI-XIV)""",
        more_info_link="https://ro.wikipedia.org/wiki/Arhitectur%C4%83_romanic%C4%83",
    )
    add_or_update_style_details(
        style_name="Romanian Revival Architecture",
        period_info="Aproximativ 1885-1930",
        region_info="România",
        more_info_text="""Stilul neoromânesc este un curent arhitectural specific României, care a căutat să creeze o arhitectură națională prin reinterpretarea elementelor din arhitectura medievală și tradițională românească (în special cea brâncovenească). Exemple definitorii, majoritatea în București, includ:
    \n- Școala Centrală, București, România (extinsă de Ion Mincu, 1890)
    \n- Bufetul de la Șosea (azi restaurantul 'Doina'), București, România (proiectat de Ion Mincu, 1892)""",
        more_info_link="https://ro.wikipedia.org/wiki/Arhitectura_neoromânească",
    )
    add_or_update_style_details(
        style_name="Russian Baroque Architecture",
        period_info="Sfârșitul sec. XVII - Mijlocul sec. XVIII",
        region_info="Rusia",
        more_info_text=""""Barocul rusesc combină elemente ale barocului vest-european cu tradițiile arhitecturale rusești. Se distinge prin culori vibrante, cupole aurite și o ornamentație bogată. Exemple ale barocului rusescsunt:
    \n- Palatul de Iarnă, Sankt Petersburg, Rusia (construit între 1754-1762)
    \n- Marele Palat de la Peterhof, lângă Sankt Petersburg, Rusia (reconstruit între 1747-1755)""",
        more_info_link="https://en.wikipedia.org/wiki/Russian_Baroque",
    )
    add_or_update_style_details(
        style_name="Russian Revival architecture",
        period_info="Mijlocul sec. XIX - Începutul sec. XX",
        region_info="Rusia",
        more_info_text="""Stilul neorus a fost o mișcare romantică ce a reînviat elemente din arhitectura rusă pre-petrină (dinainte de Petru cel Mare). Este un stil pitoresc și decorativ, inspirat de bisericile și palatele medievale rusești. Exemple iconice sunt:
    \n- Biserica Rusă 'Sf. Nicolae', București, România (sfințită în 1909)
    \n- Catedrala Sfântul Vasile, Moscova, Rusia (deși anterioară, este inspirația majoră a stilului, 1555-1561)
    \n- Biserica Mântuitorului pe Sângele Vărsat, Sankt Petersburg, Rusia (finalizată în 1907)""",
        more_info_link="https://en.wikipedia.org/wiki/Russian_Revival_architecture",
    )
    add_or_update_style_details(
        style_name="Sicilian Baroque Architecture",
        period_info="Sec. XVII - Sec. XVIII",
        region_info="Sicilia, Italia",
        more_info_text="""Barocul sicilian este o variantă distinctă a stilului baroc, dezvoltată după marele cutremur din 1693. Se caracterizează prin fațade curbe, balcoane cu balustrade din fier forjat și decorațiuni sculpturale flamboiante, inclusiv măști și putti. Exemple includ:
    \n- Catedrala din Noto, Italia (reconstruită începând cu începutul sec. XVIII)
    \n- Bazilica della Collegiata, Catania, Italia (finalizată în 1768)""",
        more_info_link="https://ro.wikipedia.org/wiki/Baroc_sicilian",
    )
    add_or_update_style_details(
        style_name="Spanish Colonial Architecture",
        period_info="Sec. XVI - Sec. XIX",
        region_info="Colonii spaniole din America și Filipine",
        more_info_text="""Arhitectura colonială spaniolă a adaptat stilurile baroc și renascentist spaniole la climatul și materialele din Lumea Nouă. Se caracterizează prin curți interioare (patio), ziduri groase, acoperișuri din țiglă ceramică și balcoane din lemn.Exemple reprezentative sunt:
    \n- Catedrala Metropolitană din Mexico City, Mexic (construită între 1573-1813)
    \n- Misiunea San Xavier del Bac, lângă Tucson, Arizona, SUA (finalizată în 1797)""",
        more_info_link="https://en.wikipedia.org/wiki/Spanish_Colonial_architecture",
    )
    add_or_update_style_details(
        style_name="Spanish Renaissance Architecture",
        period_info="Sfârșitul sec. XV - Sec. XVI",
        region_info="Spania",
        more_info_text="""Renașterea spaniolă a combinat influențe gotice, italiene și maure, având mai multe faze. Clădiri de referință sunt:
    \n- Mănăstirea El Escorial, Spania (stil herrerian, 1563-1584)
    \n- Palatul lui Carol al V-lea, Granada, Spania (început în 1527)""",
        more_info_link="https://en.wikipedia.org/wiki/Spanish_Renaissance_architecture",
    )
    add_or_update_style_details(
        style_name="Tudor Revival architecture",
        period_info="A doua jumătate a sec. XIX - Începutul sec. XX",
        region_info="Marea Britanie și Imperiul Britanic",
        more_info_text="""Stilul Tudor Revival (sau Mock Tudor) reinterpretează arhitectura vernaculară engleză din perioada Tudor. Se caracterizează prin fațade cu paiantă (half-timbering), acoperișuri abrupte, ferestre cu geamuri mici și coșuri înalte, decorative. Exemple notabile sunt:
    \n- Magazinul Liberty, Londra, Marea Britanie (deschis în 1924)
    \n- Cragside, Northumberland, Marea Britanie (remodelat masiv între 1869-1882)""",
        more_info_link="https://en.wikipedia.org/wiki/Tudor_Revival_architecture",
    )
    add_or_update_style_details(
        style_name="Venetian Gothic Architecture",
        period_info="Sec. XIV - Sec. XV",
        region_info="Veneția, Italia",
        more_info_text="""Goticul venețian este un stil unic care a fuzionat goticul european cu influențe bizantine și maure, reflectând legăturile comerciale ale Veneției. Se remarcă prin loggii delicate, arcade ogivale trilobate și fațade bogat decorate. Exemple sunt:
    \n- Palatul Dogilor, Veneția, Italia (fațadele principale construite între 1340-1450)
    \n- Ca' d'Oro (Palatul de Aur), Veneția, Italia (construit între 1428-1430)""",
        more_info_link="https://en.wikipedia.org/wiki/Venetian_Gothic_architecture",
    )
    # Exemplu de utilizator pentru testare
    # test_user_added, test_user_msg = add_user_db("testuser", "password123")
    # if test_user_added:
    #     print(f"Test user 'testuser' added successfully.")
    # elif "există deja" in test_user_msg:
    #     print(f"Test user 'testuser' already exists.")
    # else:
    #     print(f"Failed to add test user: {test_user_msg}")

    print("Detalii de stil exemplu adăugate/actualizate.")
