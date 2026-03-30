FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY server/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy environment code into a named folder to preserve python package structure
COPY . /app/email_triage_env/

ENV PYTHONPATH=/app

# Expose port (7860 for HF Spaces)
EXPOSE 7860

# Run the server
CMD ["uvicorn", "email_triage_env.server.app:app", "--host", "0.0.0.0", "--port", "7860"]
