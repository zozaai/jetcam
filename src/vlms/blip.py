from transformers import BlipProcessor, BlipForConditionalGeneration
import cv2
import os

current_dir = os.path.dirname(__file__)
image_path = os.path.join(os.path.dirname(current_dir), 'pics', 'test.jpg')

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

raw_image = cv2.imread('./test.jpg').astype('uint8')
raw_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)

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
prompt = "the image shows "
inputs = processor(raw_image, return_tensors="pt")

out = model.generate(**inputs, **gen_kwargs)
print("first out: ", processor.decode(out[0], skip_special_tokens=True))

