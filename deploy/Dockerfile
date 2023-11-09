FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /app

COPY . .

RUN pip install -r requirements.txt

EXPOSE 80/tcp

ENTRYPOINT [ "python3", "app.py", "--port", "80", "--device", "cpu" ]
