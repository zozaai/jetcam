"""
source:
https://huggingface.co/google/pix2struct-textcaps-base#using-the-model
"""

from transformers import Pix2StructForConditionalGeneration, Pix2StructProcessor
import os
import cv2

current_dir = os.path.dirname(__file__)
image_path = os.path.join(os.path.dirname(current_dir), 'pics', 'test.jpg')

image = cv2.imread('./test.jpg').astype('uint8')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

model = Pix2StructForConditionalGeneration.from_pretrained("google/pix2struct-textcaps-base")
processor = Pix2StructProcessor.from_pretrained("google/pix2struct-textcaps-base")

# image only
inputs = processor(images=image, return_tensors="pt")

predictions = model.generate(**inputs)
print(processor.decode(predictions[0], skip_special_tokens=True))