import os
import requests
from duckduckgo_search import DDGS
from tqdm import tqdm
from PIL import Image, ImageFilter
import random

ddgs = DDGS()
OUTPUT_DIR = "data/none"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Realistic queries mimicking handheld phone images but without phones
queries = [
    "hand holding remote",
    "hand holding calculator",
    "hand holding cup",
    "hand holding wallet",
    "hand holding book",
    "hand holding mouse",
    "hand holding pen",
    "hand holding sunglasses",
    "hand holding credit card",
    "hand holding object indoors"
]

# Total images per query
IMAGES_PER_QUERY = 10

# image corruption/blur to simulate phone photo quality
def apply_realistic_effects(img):
    # Randomly apply one or more effects
    if random.random() < 0.4:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))
    if random.random() < 0.3:
        img = img.resize((random.randint(200, 400), random.randint(300, 500))).resize((224, 224))
    return img

def download_images(query, output_dir):
    results = ddgs.images(query, max_results=IMAGES_PER_QUERY)
    for i, result in enumerate(results):
        try:
            image_url = result['image']
            r = requests.get(image_url, timeout=10)
            ext = image_url.split(".")[-1].split("?")[0].lower()
            if ext not in ['jpg', 'jpeg', 'png']:
                ext = 'jpg'
            filename = f"{query.replace(' ', '_')}_{i}.{ext}"
            path = os.path.join(output_dir, filename)
            with open(path, 'wb') as f:
                f.write(r.content)

            # Open and optionally apply noise/blur
            try:
                img = Image.open(path).convert('RGB')
                img = apply_realistic_effects(img)
                img.save(path)
            except Exception as img_err:
                print(f"Error processing image {filename}: {img_err}")

        except Exception as e:
            print(f"Failed to download from {image_url}: {e}")

if __name__ == "__main__":
    for query in tqdm(queries, desc="Downloading none images"):
        download_images(query, OUTPUT_DIR)
