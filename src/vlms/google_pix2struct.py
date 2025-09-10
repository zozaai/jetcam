"""
source:
https://huggingface.co/google/pix2struct-textcaps-base#using-the-model
"""

from transformers import Pix2StructForConditionalGeneration, Pix2StructProcessor
import os
import cv2

current_dir = os.path.dirname(__file__)
image_path = os.path.join(os.path.dirname(current_dir), 'pics', 'test.jpg')

image = cv2.imread(image_path).astype('uint8')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

model = Pix2StructForConditionalGeneration.from_pretrained("google/pix2struct-textcaps-base")
processor = Pix2StructProcessor.from_pretrained("google/pix2struct-textcaps-base")

# image only
inputs = processor(images=image, return_tensors="pt")

predictions = model.generate(**inputs,
                             max_length = 100, min_length = 50, num_beams=4, no_repeat_ngram_size=3, temperature=0.9, early_stopping=True,
                             top_p= 0.9, top_k= 50,  
                            #  max_new_tokens=48,          # length
                            # num_beams=4,                # beam search (higher = more thorough, slower)
                            # length_penalty=1.0,         # >1 favors shorter, <1 favors longer (beam only)
                            # no_repeat_ngram_size=3,     # reduce repetition
                            # do_sample=True,             # turn on sampling instead of greedy/beam
                            # temperature=0.7,            # randomness (lower = more deterministic)
                            # top_p=0.9,                  # nucleus sampling
                            # top_k=50,                   # or use k-sampling
                            # repetition_penalty=1.1,     # discourage repeats
                            # early_stopping=True,        # stop when all beams hit EOS
                            )
print(processor.decode(predictions[0], skip_special_tokens=True))