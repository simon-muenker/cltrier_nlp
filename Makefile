

.PHONY: install
install:
	@poetry install
	@poetry run pre-commit install


.PHONY: check
check:
	@poetry check --lock
	@poetry run pre-commit run -a
	@poetry run mypy


.PHONY: test
test:
	poetry run pytest


.PHONY: deploy
publish:
	@poetry build
	@poetry publish


.PHONY: docs
docs:
	mkdir ./docs -p
	@poetry run mkdocs build
	@poetry run mkdocs serve

.PHONY: clean
clean:
	rm -rf ./build
	rm -rf ./docs
	rm -rf ./site