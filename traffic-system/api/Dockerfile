FROM python:3.11-slim
WORKDIR /srv

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py /srv/main.py
COPY common.py /srv/common.py
COPY common_app.py /srv/common_app.py
# --- add training data into image ---
COPY data/radcom_app_train.csv /srv/data/radcom_app_train.csv
COPY data/radcom_att_train.csv /srv/data/radcom_att_train.csv


EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
