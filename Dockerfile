FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (cache layer)
COPY email_triage_env/server/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Expose port (7860 for HF Spaces)
EXPOSE 7860

# Run the FastAPI app
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
