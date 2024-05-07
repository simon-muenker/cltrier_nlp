# CLTrier NLP: academic teaching toolbox

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

## Project Structure

```   
│ 
├── Makefile                    <- Makefile containing development targets
├── README.md                   <- top-level README
├── pyproject.toml              <- package-level (poetry) configuration
├── mkdocs.yaml                 <- documentation configuration
├── .pre-commit-config.yaml     <- git pre-commit actions
│
├── cltrier_nlp                 <- root source
│   └── corpus                  <- nltk inspired corpus module
│   └── encoder                 <- huggingface auto model wrapper
│   └── trainer                 <- pytorch training algorithm
│   └── functional              <- generic helper functions
│   └── utility                 <- utility classes and types
│
├── tests                       <- unittests
│
├── examples                    <- usage/application examples
│
├── scripts                 <   - additional package building scripts
│   └── gen_docs_pages.py       <- automatic doc generation based on docstrings
│
```

## Resources

- Project Template (Data Science): <https://github.com/drivendata/cookiecutter-data-science>
- Project Template (Poetry): <https://github.com/fpgmaas/cookiecutter-poetry>