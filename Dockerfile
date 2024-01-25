FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 -y

RUN apt update && apt install -y bash \
                   git \
                   curl

RUN apt install -y python3.10 \
                   python3-pip

RUN python3.10 -m pip install torch==2.0.1
RUN python3.10 -m pip install torchvision==0.15.2
RUN python3.10 -m pip install transformers==4.31.0
RUN python3.10 -m pip install tokenizers==0.12.1
RUN python3.10 -m pip install sentencepiece==0.1.99
RUN python3.10 -m pip install shortuuid
RUN python3.10 -m pip install accelerate==0.21.0
RUN python3.10 -m pip install peft==0.4.0
RUN python3.10 -m pip install bitsandbytes==0.41.0
RUN python3.10 -m pip install pydantic==1.10.13
RUN python3.10 -m pip install markdown2[all]
RUN python3.10 -m pip install numpy
RUN python3.10 -m pip install scikit-learn==1.2.2
RUN python3.10 -m pip install gradio==3.35.2
RUN python3.10 -m pip install gradio_client==0.2.9
RUN python3.10 -m pip install requests
RUN python3.10 -m pip install httpx==0.24.0
RUN python3.10 -m pip install uvicorn
RUN python3.10 -m pip install fastapi
RUN python3.10 -m pip install einops==0.6.1
RUN python3.10 -m pip install einops-exts==0.0.4
RUN python3.10 -m pip install timm==0.6.13
RUN python3.10 -m pip install wandb