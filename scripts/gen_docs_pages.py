"""
Generate the code reference pages and navigation.
src: https://mkdocstrings.github.io/recipes/
"""

from pathlib import Path

import mkdocs_gen_files

nav = mkdocs_gen_files.Nav()

root = Path(__file__).parent.parent
src = root


for path in sorted(src.rglob("cltrier_nlp/**/*.py")):

    module_path = path.relative_to(src).with_suffix("")
    doc_path = path.relative_to(src).with_suffix(".md")

    parts = tuple(module_path.parts)

    if parts[-1] == "__init__":
        parts = parts[:-1]
        doc_path = doc_path.with_name("index.md")

    elif parts[-1] == "__main__":
        continue

    if parts:
        nav[parts] = doc_path.as_posix()

    with mkdocs_gen_files.open(doc_path, "w") as fd:
        ident = ".".join(parts)
        fd.write(f"::: {ident}")

    mkdocs_gen_files.set_edit_path(doc_path, path.relative_to(root))


with mkdocs_gen_files.open("index.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())


print(mkdocs_gen_files.edit_paths)
print(nav.__dict__)
