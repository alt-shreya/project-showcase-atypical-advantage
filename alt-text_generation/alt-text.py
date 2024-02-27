import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

# Initialize image captioning model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def generate_alternative_text(image_url):
    # Download the image
    image = Image.open(requests.get(image_url, stream=True).raw)

    # Process the image and generate alternative text
    inputs = processor(images=image, return_tensors="pt")
    out = model.generate(**inputs)
    description = processor.decode(out[0], skip_special_tokens=True)

    return description

# replace the url with the image to be captioned
image_url = "https://t4.ftcdn.net/jpg/01/77/47/67/360_F_177476718_VWfYMWCzK32bfPI308wZljGHvAUYSJcn.jpg"
alt_text = generate_alternative_text(image_url)
print("Alternative Text:", alt_text)