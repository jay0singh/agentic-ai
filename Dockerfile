FROM python:3.13-slim

WORKDIR /app

COPY app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ .

EXPOSE 8000 8501

# Default command runs the API; the compose ui service overrides this with streamlit.
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
