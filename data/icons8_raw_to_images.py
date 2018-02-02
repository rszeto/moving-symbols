import os
from PIL import Image

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    raw_images_root = os.path.join(script_dir, 'icons8_raw')
    final_images_root = os.path.join(script_dir, 'icons8')
    final_size = (28, 28)
    resample_strategy = Image.NEAREST

    for category in os.listdir(raw_images_root):
        if not os.path.isdir(os.path.join(raw_images_root, category)):
            continue

        # Make training/testing directories
        if not os.path.exists(os.path.join(final_images_root, 'training', category)):
            os.makedirs(os.path.join(final_images_root, 'training', category))
        if not os.path.exists(os.path.join(final_images_root, 'testing', category)):
            os.makedirs(os.path.join(final_images_root, 'testing', category))

        # Process each image for the category
        category_image_filenames = os.listdir(os.path.join(raw_images_root, category))
        for i, filename in enumerate(category_image_filenames):
            split = 'training' if i % 2 == 0 else 'testing'
            raw_image_path = os.path.join(raw_images_root, category, filename)
            final_image_path = os.path.join(final_images_root, split, category, filename)
            image = Image.open(raw_image_path)
            small_image = image.resize(final_size, resample_strategy)
            small_image.save(final_image_path)

if __name__ == '__main__':
    main()