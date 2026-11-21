# Triage assistant: docker build -t severity-triage . && docker run -p 8000:8000 severity-triage
FROM python:3.12-slim
WORKDIR /srv
COPY requirements.txt ./reqs/requirements.txt
COPY app/requirements.txt ./reqs/app/requirements.txt
RUN pip install --no-cache-dir -r reqs/requirements.txt -r reqs/app/requirements.txt
COPY src ./src
COPY app ./app
COPY prompts ./prompts
COPY models ./models
ENV LLM_PROVIDER=ollama LLM_MODEL=llama3.1:8b OLLAMA_URL=http://ollama:11434 TRIAGE_LOG_DB=/srv/data/triage_log.db
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
