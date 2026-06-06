# FROM python:3.11-slim

# WORKDIR /app

# COPY . .

# RUN pip install -r requirements.txt

# EXPOSE 8003

# CMD ["uvicorn","app.main:app","--host","0.0.0.0","--port","8003"]

# ==============================================================
# Dockerfile  (sits at root: Airline_Satisfaction_Prediction/)
#
# BUILD:
#   docker build -t airline-mlmodel .
#
# RUN:
#   docker run -p 8003:8003 airline-mlmodel
#
# OPEN:
#   http://localhost:8003/docs
# ==============================================================

FROM python:3.11-slim

# All files inside the container live under /app
WORKDIR /app

# ── Install packages ──────────────────────────────────────────
# Copy requirements first so Docker can cache this layer.
# If requirements.txt hasn't changed, Docker skips reinstalling.
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy project folders ──────────────────────────────────────
COPY app/          app/
COPY training/     training/
COPY data/         data/
COPY saved_models/ saved_models/


# ── Start the API ─────────────────────────────────────────────
EXPOSE 8003

# app.main  →  app/main.py  →  the FastAPI app object called "app"
# --host 0.0.0.0  = accept connections from outside the container
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8003"]


