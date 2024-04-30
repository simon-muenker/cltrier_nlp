# Project Repository: NLP Course (M.Sc. NLP, University Trier)

## Usage

### Install

```bash
pip install cltrier_nlp
```

## Development

### Install

The project is managed by Poetry, a dependency management and packaging library. Please set up a local version according to the [official installation guidelines](https://python-poetry.org/docs/). When finished, install the local repository as follows:

```bash
# install package dependencies
poetry install

# add pre-commit to git hooks
poetry run pre-commit install  
```

### Tests

```bash
poetry run pytest
```

### Linting

```bash
poetry run pre-commit run --all-files
```