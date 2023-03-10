FROM python:3.9

WORKDIR /

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

ENTRYPOINT ["uvicorn"]
CMD ["main:app", "--host", "0.0.0.0", "--port", "5000"]