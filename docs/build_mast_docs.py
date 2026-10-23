from pathlib import Path
import shutil

source = Path("_build/mast_source")
shutil.rmtree(source, ignore_errors=True)
source.mkdir(parents=True)
shutil.copytree("mast", source / "mast", ignore=shutil.ignore_patterns("api"))
shutil.copytree("_static", source / "_static")
shutil.copytree("_templates", source / "_templates")
(source / "index.rst").write_text(
    "MAST Documentation\n"
    "==================\n\n"
    ".. toctree::\n"
    "   :hidden:\n\n"
    "   mast/mast\n"
)