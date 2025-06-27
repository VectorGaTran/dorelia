from PIL import Image, UnidentifiedImageError, ImageFile, ImageOps 
import os
import numpy as np
from urllib.parse import unquote

ImageFile.LOAD_TRUNCATED_IMAGES = True

# --- Configurație ---
INPUT_DIR = "personal test"
OUTPUT_DIR = "pula"
TARGET_SIZE = (224, 224)

def ensure_dir(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

# --- Bucla principală a procesului ---
def preprocess_images(root_input_dir, root_output_dir, target_size):
    ensure_dir(root_output_dir)
    total_images_processed = 0
    error_count = 0
    skipped_already_exists = 0

    print(f"Date de intrare din: {os.path.abspath(root_input_dir)}")
    print(f"Date de ieșire din: {os.path.abspath(root_output_dir)}")
    print(f"Dimensiunea dorită pentru imagini: {target_size}")

    for style_folder_name in os.listdir(root_input_dir):
        style_input_path = os.path.join(root_input_dir, style_folder_name)
        style_output_path = os.path.join(root_output_dir, style_folder_name)

        if not os.path.isdir(style_input_path):
            continue

        ensure_dir(style_output_path)
        print(f"Se procesează imaginile legate de stilul: {style_folder_name}")
            
        current_style_images_processed = 0

        image_filenames_in_style_folder = []
        try:
            image_filenames_in_style_folder = os.listdir(style_input_path)
        except Exception as e_listdir:
            error_count +=1
            continue

        for image_filename_original in image_filenames_in_style_folder:
            if not image_filename_original.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp')):
                continue

            image_filename_decoded = unquote(image_filename_original)
            input_image_path = os.path.join(style_input_path, image_filename_decoded)
            output_image_path = os.path.join(style_output_path, image_filename_decoded)
                
            if os.path.exists(output_image_path):
                skipped_already_exists +=1
                total_images_processed +=1 
                current_style_images_processed +=1
                continue

            try:
                if not os.path.exists(input_image_path):
                    input_image_path_fallback = os.path.join(style_input_path, image_filename_original)
                    if not os.path.exists(input_image_path_fallback):
                        error_count +=1
                        continue
                    else:
                        input_image_path = input_image_path_fallback
                        output_image_path = os.path.join(style_output_path, image_filename_original)
                        if os.path.exists(output_image_path):
                            skipped_already_exists +=1
                            total_images_processed +=1
                            current_style_images_processed +=1
                            continue

                with Image.open(input_image_path) as img:
                    img_oriented = ImageOps.exif_transpose(img)
                    
                    img_rgb = img_oriented.convert('RGB') # asigură 3 canale
                    
                img_resized = img_rgb.resize(target_size, Image.Resampling.LANCZOS) # redimensionare
                img_resized.save(output_image_path)
                    
                total_images_processed += 1
                current_style_images_processed += 1

            except FileNotFoundError:
                print(f"    FileNotFoundError pentru: {input_image_path} (nume original: {image_filename_original})")
                error_count +=1
            except UnidentifiedImageError:
                print(f"    A fost sărită imagine coruptă/neidentificată: {input_image_path}")
                error_count +=1
            except Exception as e:
                print(f"    Eroare în procesarea imaginii {input_image_path} (original: {image_filename_original}): {e}")
                error_count +=1
            
        if current_style_images_processed > 0:
            print(f"    S-a terminat procesarea și salvarea imaginilor pentru stilul{style_folder_name}: {current_style_images_processed} .")

    print(f"\n--- Procesarea s-a terminat ---")
    print(f"Total imagini procesate: {total_images_processed}")
    print(f"S-a sărit peste {skipped_already_exists} imagini care erau deja în date de ieșire.")
    print(f"Total erori: {error_count}")

if __name__ == "__main__":
    if not os.path.isdir(INPUT_DIR):
        print(f"Eroare: directorul cu date de intrare '{INPUT_DIR}' nu a fost găsit.")
        print(f"Structura dorită {INPUT_DIR}/Nume_stil/imagine.jpg")
        print("Programul nu va continua.")
    else:
        preprocess_images(INPUT_DIR, OUTPUT_DIR, TARGET_SIZE)