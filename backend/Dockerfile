FROM python:3.12-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt
ARG INSTALL_MODELS=false
COPY backend/requirements-models.txt /app/backend/requirements-models.txt
RUN if [ "$INSTALL_MODELS" = "true" ]; then pip install --no-cache-dir torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu && pip install --no-cache-dir -r backend/requirements-models.txt; fi
COPY backend/quantara /app/backend/quantara
COPY backend/app.py /app/backend/app.py
RUN useradd --create-home quantara && mkdir /app/runtime && chown quantara:quantara /app/runtime
USER quantara
ENV PYTHONPATH=/app/backend
EXPOSE 8000
CMD ["uvicorn", "app:app", "--app-dir", "backend", "--host", "0.0.0.0", "--port", "8000"]
