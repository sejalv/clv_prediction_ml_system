.DEFAULT_GOAL := help

# Override if Poetry is not on PATH, e.g. POETRY=/opt/poetry/bin/poetry make setup
POETRY ?= poetry

#help:				@ list available goals
.PHONY: help
help:
	@grep -E '[a-zA-Z\.\-]+:.*?@ .*$$' $(MAKEFILE_LIST)| sort | tr -d '#'  | awk 'BEGIN {FS = ":.*?@ "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

#setup:				@ install dependencies configured in pyproject.toml
.PHONY: setup
setup:
	@echo " install dependencies"
	@command -v "$(POETRY)" >/dev/null 2>&1 || { \
		echo ""; \
		echo "make setup: '$(POETRY)' not found."; \
		echo "Install Poetry, then retry (ensure its bin dir is on PATH):"; \
		echo "  https://python-poetry.org/docs/#installation"; \
		echo "  macOS:   brew install poetry"; \
		echo "  or:      pipx install poetry"; \
		echo ""; \
		echo "Docker-only workflow: skip setup; use 'make run', 'make test', 'make train'."; \
		echo ""; \
		exit 1; \
	}
	$(POETRY) config virtualenvs.create true
	$(POETRY) config virtualenvs.in-project true
	$(POETRY) install --no-interaction

#test:				@ run test with docker
.PHONY: test
test:
	@echo " running tests"
	docker-compose down
	docker-compose up --build test
	docker-compose down

#run:				@ run application in docker
.PHONY: run
run:
	@echo " running service"
	docker-compose down
	docker-compose up --build app


#train:				@ run model training procedure (to be implemented)
.PHONY: train
train:
	@echo " training model"
	docker-compose run --rm \
		-e TRAIN_GOLDEN_MAX_ROWS \
		-e TRAIN_RANDOM_SEED \
		-e SQLITE_PATH \
		-e MODEL_DIR \
		app python -m app.training
