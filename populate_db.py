
import os
from db_utils import DBUtils

def populate_database():
    """
    Populates the SQLite database with image paths and labels.
    """
    db = DBUtils(db_name='user_data.db')
    print("Ensuring 'styles' table exists...")
    db.create_table()

    # --- IMPORTANT ---
    # Modify this list to include your actual image data.
    # Each tuple should contain the absolute path to an image and its label.
    #
    # For example:
    # images_to_add = [
    #     ('/path/to/your/images/gothic/image1.jpg', 'Gothic'),
    #     ('/path/to/your/images/gothic/image2.png', 'Gothic'),
    #     ('/path/to/your/images/modern/buildingA.jpg', 'Modern'),
    # ]
    
    images_to_add = []

    # Example of how you might automatically populate this list
    # if your images are organized in labeled subdirectories.
    
    base_image_directory = '/path/to/your/images' # <--- CHANGE THIS
    
    if base_image_directory == '/path/to/your/images':
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!! PLEASE UPDATE 'base_image_directory' IN THIS SCRIPT !!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        return

    for label in os.listdir(base_image_directory):
        label_dir = os.path.join(base_image_directory, label)
        if os.path.isdir(label_dir):
            for image_file in os.listdir(label_dir):
                if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(label_dir, image_file)
                    images_to_add.append((image_path, label))


    print(f"Found {len(images_to_add)} images to add to the database.")

    for filename, label in images_to_add:
        # Check if the image already exists in the DB
        if not db.get_image_by_filename(filename):
            print(f"Inserting: {filename} -> {label}")
            db.insert_image(filename, label)
        else:
            print(f"Skipping, already in DB: {filename}")

    db.close_connection()
    print("Database population complete.")

if __name__ == '__main__':
    populate_database()
