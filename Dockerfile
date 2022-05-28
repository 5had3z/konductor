FROM alpine:3.16.0

# Basic Python Setup
ENV PYTHONUNBUFFERED=1
RUN apk add --update --no-cache python3 && \
    ln -sf python3 /usr/bin/python && \
    python3 -m ensurepip && \
    pip3 install --no-cache --upgrade pip setuptools

# Install Dependencies
RUN pip3 install torch==1.11.0 --extra-index-url https://download.pytorch.org/whl/cpu
RUN pip3 install dash
