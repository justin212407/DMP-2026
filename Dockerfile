FROM ghcr.io/huggingface/trl-latest-gpu:latest

WORKDIR /workspace

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY dpo_train.py .
COPY accelerate_config.yaml .

CMD ["python", "dpo_train.py", "--help"]
