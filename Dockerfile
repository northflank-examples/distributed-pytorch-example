FROM pytorch/pytorch:2.9.1-cuda12.6-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

WORKDIR /workspace

COPY train.py .
COPY entrypoint.sh .

RUN chmod +x entrypoint.sh

ENV REPLICAS=2
ENV MASTER_PORT=29500
ENV NPROC_PER_NODE=1

ENTRYPOINT ["./entrypoint.sh"]
