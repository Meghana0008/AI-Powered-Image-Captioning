# AI-Powered-Image-Captioning

## Step 1: Import Required Libraries
```python
import json
import requests
from pycocotools.coco import COCO
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from io import BytesIO
from google.colab import files
from transformers import BlipProcessor, BlipForConditionalGeneration
import zipfile
```

## Step 2: Download and Extract COCO Annotations
```python
# Define the URL for the COCO dataset annotations file
annotations_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

# Download the annotations file
annotations_file = "/content/annotations_trainval2017.zip"
response = requests.get(annotations_url)
with open(annotations_file, "wb") as f:
    f.write(response.content)

# Unzip the annotations file
with zipfile.ZipFile(annotations_file, "r") as zip_ref:
    zip_ref.extractall("/content/coco_annotations")
```

## Step 3: Load COCO Dataset
```python
# Load the annotations into the COCO API
annotations_file_path = "/content/coco_annotations/annotations/captions_train2017.json"
coco = COCO(annotations_file_path)

# Display dataset information
print(f"COCO Dataset contains {len(coco.dataset['images'])} images")
print(f"COCO Dataset contains {len(coco.dataset['annotations'])} annotations")
```

## Step 4: Fetch and Display an Image with Captions
```python
# Get the ID of the first image
image_id = coco.getImgIds()[0]
image_info = coco.loadImgs(image_id)[0]

# Fetch the image
image_url = image_info['coco_url']
image_data = requests.get(image_url).content
image = Image.open(BytesIO(image_data))

# Fetch the annotations (captions)
annotation_ids = coco.getAnnIds(imgIds=image_id)
annotations = coco.loadAnns(annotation_ids)

# Display the image and captions
plt.imshow(np.asarray(image))
plt.axis('off')
plt.title('Captions: ' + ' | '.join([ann['caption'] for ann in annotations]), fontsize=12)
plt.show()
```

## Step 5: Load BLIP Image Captioning Model
```python
# Load BLIP processor and model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
```

## Step 6: Upload an Image
```python
# Function to upload an image using Google Colab

def upload_image():
    uploaded = files.upload()
    if len(uploaded) == 0:
        print("No image uploaded!")
        return None
    image_path = list(uploaded.keys())[0]
    print(f"Image uploaded: {image_path}")
    return image_path
```

## Step 7: Generate Caption using BLIP
```python
# Function to generate a caption from the uploaded image
def generate_caption(image_path):
    # Open and process the image
    raw_image = Image.open(image_path).convert("RGB")

    # Preprocess the image for BLIP
    inputs = processor(raw_image, return_tensors="pt")

    # Generate the caption
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)

    return caption
```

## Step 8: Display Image with Generated Caption
```python
# Main function to handle image upload and captioning
def main():
    image_path = upload_image()
    if image_path:
        # Generate caption for the uploaded image
        caption = generate_caption(image_path)

        # Display the image and its caption
        img = Image.open(image_path)
        plt.imshow(np.asarray(img))
        plt.axis('off')
        plt.title(caption, fontsize=12)
        plt.show()

# Run the main function
if __name__ == "__main__":
    main()
