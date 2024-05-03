"""
Generate the code reference pages and navigation.
src: https://mkdocstrings.github.io/recipes/
"""

from pathlib import Path
import re

import mkdocs_gen_files

nav = mkdocs_gen_files.Nav()

PACKAGE: str = "cltrier_nlp"

ROOT: Path = Path(__file__).parent.parent
SRC: Path = ROOT / PACKAGE

INDEX_PAGE: Path = ROOT / "README.md"
NEW_INDEX_TITLE: str = "Getting started"

for path in sorted(SRC.rglob("*.py")):

    module_path = path.relative_to(SRC).with_suffix("")
    doc_path = path.relative_to(SRC).with_suffix(".md")

    parts = tuple(module_path.parts)

    if parts[-1] == "__init__":
        parts = parts[:-1]
        doc_path = doc_path.with_name("index.md")

    elif parts[-1] == "__main__":
        continue

    if parts:
        nav[parts] = doc_path.as_posix()

    with mkdocs_gen_files.open(doc_path, "w") as fd:
        if parts:
            ident = ".".join(parts)
            fd.write(f"::: {PACKAGE}.{ident}")

    mkdocs_gen_files.set_edit_path(doc_path, path.relative_to(ROOT))


with mkdocs_gen_files.open("index.md", "a") as nav_file:
    nav_file.writelines(
        re.sub(r"^#\s.*\n", f"# {NEW_INDEX_TITLE}\n", open(INDEX_PAGE).read())
    )
    nav_file.writelines("\n## Sitemap\n")
    nav_file.writelines(nav.build_literate_nav())
