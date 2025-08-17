import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import cv2

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg' 
# raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')

raw_image = cv2.imread('./test.jpg').astype('uint8')

gen_kwargs = dict(
    do_sample=False,          # no sampling
    num_beams=6,              # 4–8 is common
    max_new_tokens=120,       # don’t let it ramble
    min_new_tokens=60,        # ensure enough detail
    no_repeat_ngram_size=3,   # stop phrase loops
    length_penalty=0.95,      # slightly prefer concise
    early_stopping=True,   

)

# conditional image captioning
prompt = "explain the image with as much detail as possible"
inputs = processor(raw_image, return_tensors="pt")


out = model.generate(**inputs, **gen_kwargs)
print("first out: ", processor.decode(out[0], skip_special_tokens=True))

