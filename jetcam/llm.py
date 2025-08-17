# jetcam/llm.py
from dotenv import load_dotenv
import os
from openai import OpenAI

load_dotenv(override=True)
print(os.environ.get("OPENAI_API_KEY"))

client = OpenAI()
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "system",
            "content": (
                "You are VisionBot: an onboard computer-vision assistant running on an NVIDIA Jetson Nano "
                "with a live webcam feed. Your task is to continuously analyze each camera frame and provide "
                "concise, structured descriptions of everything you see. "
                "Focus on identifying objects, people, actions, spatial relationships, and environmental context. "
                "Output your observations in JSON form with keys: "
                "`timestamp` (ISO 8601), `objects` (list of {name, bbox}), "
                "`actions` (list of verbs), and `scene_description` (one-two sentences)."
            )
        },
        {
            "role": "user",
            "content": "Here's the current frame from the webcam: <https://github.com/zozaai/jetcam/blob/main/docs/logo.jpeg>"
        }
    ]
)

print(response.choices[0].message.content)
