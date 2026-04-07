FROM python:3.10-slim as base

WORKDIR /app

ENV POETRY_HOME="/opt/poetry"
ENV POETRY_VERSION="1.8.2"
ENV PYTHONPATH=/app

RUN apt-get update && apt-get install --no-install-recommends -y \
    curl \
    && rm -rf /var/lib/apt/lists/*
RUN curl -sSL https://install.python-poetry.org | python3 - \
    && ln -s /opt/poetry/bin/poetry /usr/local/bin/poetry \
    && poetry config virtualenvs.create false

COPY poetry.lock* pyproject.toml ./
RUN poetry install --no-cache --no-interaction --no-root --without dev

COPY app ./app
COPY database.sqlite ./database.sqlite

ENTRYPOINT [ "poetry", "run" ]

FROM base as test

RUN poetry install --no-cache --no-interaction --no-root

COPY test ./test

FROM base as app

EXPOSE 8080

CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8080"]
