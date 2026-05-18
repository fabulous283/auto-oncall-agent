FROM mcr.microsoft.com/devcontainers/python:1-3.12-bookworm

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY pyproject.toml README.md ./
COPY app ./app
COPY static ./static
COPY mcp_servers ./mcp_servers
COPY aiops-docs ./aiops-docs

RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -e .

EXPOSE 9900 8003 8004

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "9900"]
