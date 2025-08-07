<p align="center">
  <a href="https://excalidraw.com/#json=FcO55BsQn51s2Pqqt5rrK,oh1x03sJwQH__qTI1Zd1tw">
    <img src="docs/block-diagram.svg" alt="Jet Voice Block Diagram" width="125%" />
  </a>
</p>


# gitcam
🤖 JetSmartCam — Jetson Nano + Camera + VLLM + TTS = real-time scene talker! 📷🧠🔊


## 📝 To-Do
- [x] Openai calling demo code
- [ ] [VLM Module]()
- [ ] [Text To Speech]()
- [ ] [Docker Image]()
- [ ] [Test]()
  - [ ] [Test VLM]()
  - [ ] [Test TTS]()


# How to setup
Copy .env.example to .env, then set environment variables

## set .env
```bash
OPENAI_API_KEY="Place you openai api key here"
```

# How test openai calling
```
python -m jetcam.llm
```
