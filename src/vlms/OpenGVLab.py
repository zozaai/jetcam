"""
source:
https://internvl.readthedocs.io/en/latest/internvl3.0/deployment.html
"""
from lmdeploy import pipeline, TurbomindEngineConfig
from lmdeploy.vl import load_image

model = 'OpenGVLab/InternVL3-78B'
image = load_image('https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/tests/data/tiger.jpeg')
pipe = pipeline(model, backend_config=TurbomindEngineConfig(session_len=8192, tp=4), chat_template_config=ChatTemplateConfig(model_name='internvl2_5'))
response = pipe(('describe this image', image))
print(response.text)