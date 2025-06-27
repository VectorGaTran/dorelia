import requests
import os
import time
import re
from urllib.parse import unquote

# --- Configurație ---
BASE_DATA_DIR = "wikimedia_scraped_images"
API_URL = "https://commons.wikimedia.org/w/api.php"
HEADERS = {
    'User-Agent': 'ArchitecturalStyleResearchBot/1.0 (Student Thesis Project; contact: victortrandafir@gmail.com)'
}
REQUEST_DELAY = 1
CATEGORIES_TO_SCRAPE = {
    "Romanian Revival Architecture": [
        "Category:Craiova Archeology and History Museum",
    ]
}
AVOID_CATEGORY_KEYWORDS = [
    "typography", "interiors", "architects", "maps of",
    "people", "logos", "models of buildings", "ruins of",
    "drawings of", "paintings of", "stamps"
]
AVOID_SPECIFIC_CATEGORIES = [
]
MAX_IMAGES_PER_STYLE = 300

# --- Funcții ajutătoare ---
def ensure_dir(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

def sanitize_filename(filename):
    filename = unquote(filename)
    filename = re.sub(r'[\\/*?:"<>|]', "", filename)
    filename = filename.replace(" ", "_")
    return filename

def download_image_from_api(file_page_title, save_dir, style_name_for_filename):
    params = {
        "action": "query",
        "format": "json",
        "titles": file_page_title,
        "prop": "imageinfo",
        "iiprop": "url|size|mime",
        "iiurlwidth": 800
    }
    try:
        response = requests.get(API_URL, params=params, headers=HEADERS)
        response.raise_for_status()
        data = response.json()
        pages = data.get("query", {}).get("pages", {})

        for page_id, page_info in pages.items():
            if "imageinfo" in page_info:
                image_info = page_info["imageinfo"][0]
                img_url = image_info.get("url")
                mime_type = image_info.get("mime", "").lower()

                if not img_url:
                    return False

                if not (mime_type.startswith("image/jpeg") or mime_type.startswith("image/png") or mime_type.startswith("image/gif")):
                    print(f"  Se sare imagine de format non-standard '{mime_type}' pentru {file_page_title}")
                    return False

                original_filename = page_info.get("title", f"{style_name_for_filename}_{int(time.time()*1000)}.jpg").replace("File:", "")
                save_filename = sanitize_filename(original_filename)
                full_save_path = os.path.join(save_dir, save_filename)

                if os.path.exists(full_save_path):
                    return False

                img_response = requests.get(img_url, headers=HEADERS, stream=True, timeout=30)
                img_response.raise_for_status()
                with open(full_save_path, 'wb') as f:
                    for chunk in img_response.iter_content(chunk_size=8192):
                        f.write(chunk)
                print(f"{save_filename} a fost descărcată cu succes")
                return True

    except requests.exceptions.RequestException as e:
        print(f"  Eroare la descărcarea imaginii {file_page_title}: {e}")
    except Exception as e_inner:
        print(f"  Eroare neașteptată cu {file_page_title}: {e_inner}")
    return False

def get_category_members(category_title, processed_categories, processed_files, download_count_for_style, style_save_dir, style_name_for_file):
    if category_title in processed_categories or download_count_for_style['count'] >= MAX_IMAGES_PER_STYLE:
        return

    print(f"Categoria procesată: {category_title} (Images downloaded for style: {download_count_for_style['count']}/{MAX_IMAGES_PER_STYLE})")
    processed_categories.add(category_title)

    if category_title in AVOID_SPECIFIC_CATEGORIES:
        print(f"  S-a sărit categoria de evitat: {category_title}")
        return
    for keyword in AVOID_CATEGORY_KEYWORDS:
        if keyword.lower() in category_title.lower():
            print(f"  S-a sărit categoria deoarece conținea cuvântul'{keyword}': {category_title}")
            return

    cmcontinue = None
    while True:
        if download_count_for_style['count'] >= MAX_IMAGES_PER_STYLE:
            print(f"  Numărul de imagini descărcate maxim a fost atins pentru stilul {style_name_for_file}.")
            break

        params = {
            "action": "query",
            "format": "json",
            "list": "categorymembers",
            "cmtitle": category_title,
            "cmlimit": "50",
            "cmtype": "file|subcat", # cere fișiere și subcategorii
            "cmprop": "ids|title|type|sortkey|timestamp|sortkeyprefix|ns" # cere namespace (ns)
        }
        if cmcontinue:
            params["cmcontinue"] = cmcontinue

        try:
            response = requests.get(API_URL, params=params, headers=HEADERS, timeout=20)
            response.raise_for_status()
            data = response.json()
            members = data.get("query", {}).get("categorymembers", [])

            for member in members:
                if download_count_for_style['count'] >= MAX_IMAGES_PER_STYLE:
                    break
                
                member_title = member.get("title")
                member_type = member.get("type")
                member_ns = member.get("ns")

                if member_type is None:
                    if member_ns == 6: 
                        member_type = "file"
                    elif member_ns == 14: 
                        member_type = "subcat"
                    else:
                        continue # se sare dacă nu distinge dacă e fișier sau subcategorie
                # Process based on determined type
                if member_type == "file":
                    if member_title not in processed_files:
                        if download_image_from_api(member_title, style_save_dir, style_name_for_file):
                            download_count_for_style['count'] += 1
                        processed_files.add(member_title)
                        time.sleep(0.1)

                elif member_type == "subcat":
                    time.sleep(REQUEST_DELAY / 2.0)
                    get_category_members(member_title, processed_categories, processed_files, download_count_for_style, style_save_dir, style_name_for_file)
                
            cmcontinue = data.get("continue", {}).get("cmcontinue")
            if not cmcontinue:
                break 
            
            time.sleep(REQUEST_DELAY)

        except requests.exceptions.Timeout:
            try:
                response = requests.get(API_URL, params=params, headers=HEADERS, timeout=30) # timeout mai lung 
                response.raise_for_status()
                data = response.json()
                members = data.get("query", {}).get("categorymembers", [])
                for member in members:
                    if download_count_for_style['count'] >= MAX_IMAGES_PER_STYLE: break
                    member_title = member.get("title")
                    member_type = member.get("type")
                    member_ns = member.get("ns")
                    if not member_title: continue
                    if member_type is None:
                        if member_ns == 6: member_type = "file"
                        elif member_ns == 14: member_type = "subcat"
                        else: continue
                    if member_type == "file":
                        if member_title not in processed_files:
                            if download_image_from_api(member_title, style_save_dir, style_name_for_file):
                                download_count_for_style['count'] += 1
                            processed_files.add(member_title)
                            time.sleep(0.1)
                    elif member_type == "subcat":
                        time.sleep(REQUEST_DELAY / 2.0)
                        get_category_members(member_title, processed_categories, processed_files, download_count_for_style, style_save_dir, style_name_for_file)
                cmcontinue = data.get("continue", {}).get("cmcontinue")
                if not cmcontinue: break

            except Exception as e_retry:
                print(f"Reîncarcare eșuată pentru categoria: {category_title}: {e_retry}")
                break 
        except requests.exceptions.RequestException as e:
            print(f"Error fetching members for category {category_title}: {e}")
            break
        except Exception as e_inner: 
            break
        
# --- Rularea principală--
if __name__ == "__main__":
    ensure_dir(BASE_DATA_DIR)

    for style_key, root_categories in CATEGORIES_TO_SCRAPE.items():
        print(f"\n===== A început scrape pentru stilul: {style_key} =====")
        style_save_path = os.path.join(BASE_DATA_DIR, style_key)
        ensure_dir(style_save_path)

        processed_categories_for_style = set()
        processed_files_for_style = set()
        images_downloaded_for_style = {'count': 0}


        for root_cat in root_categories:
            if images_downloaded_for_style['count'] < MAX_IMAGES_PER_STYLE:
                get_category_members(
                    root_cat,
                    processed_categories_for_style,
                    processed_files_for_style,
                    images_downloaded_for_style,
                    style_save_path,
                    style_key
                )
            else:
                print(f"S-a atins numărul maxim de imagini pentru stilul {style_key} înainte de procesarea tuturor categoriilor.")
                break
        
        print(f"===== S-a terminat scrape-ul pentru stilul: {style_key}. Număr imagini descărcate: {images_downloaded_for_style['count']} =====")
        if images_downloaded_for_style['count'] < MAX_IMAGES_PER_STYLE:
            print(f"Au fost găsite și descărcate mai puține imagini ({MAX_IMAGES_PER_STYLE}) admise pentru stilul {style_key}.")
        time.sleep(REQUEST_DELAY * 5) 

    print(f"Imaginile au fost salvate în: {os.path.abspath(BASE_DATA_DIR)}")