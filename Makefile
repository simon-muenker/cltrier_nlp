
.PHONY: install
install:
	@poetry install
	@poetry run pre-commit install
	@poetry run mypy --install-types


.PHONY: check
check:
	@poetry check --lock
	@poetry run pre-commit run -a
	@poetry run mypy


.PHONY: test
test:
	poetry run pytest


.PHONY: deploy
deploy:
	@poetry build
	@poetry publish
	deploy_docs

.PHONY: deploy_docs
deploy_docs:
	mkdir -p ./docs
	@poetry run mkdocs gh-deploy
	$(MAKE) clean

.PHONY: docs
docs:
	mkdir -p ./docs
	@poetry run mkdocs serve


.PHONY: clean
clean:
	rm -rf ./build
	rm -rf ./docs
	rm -rf ./site
	rm -rf ./dist
