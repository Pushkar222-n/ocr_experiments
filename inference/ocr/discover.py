"""Find the documents to process, recursively, from a directory or a zip.

chandra's own CLI globs one level deep (`input.glob("*.pdf")`), so nested folders would be
silently skipped. We walk the tree instead, and keep each file's path *relative to the input
root* so the output tree can mirror the input tree exactly.
"""
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path

from .config import IMAGE_SUFFIXES, PDF_SUFFIXES, WORK


@dataclass(frozen=True)
class Doc:
    path: Path  # absolute path to the file on disk
    rel: Path  # path relative to the input root, e.g. Batch1/sub/report.pdf
    root: Path  # the input root it was found under

    @property
    def stem(self) -> str:
        return self.path.stem

    @property
    def out_dir_rel(self) -> Path:
        """Where this doc's outputs live, relative to the run's output root.

        Mirrors the input tree, then one folder per document — chandra writes its extracted
        images beside the markdown and the markdown links to them by bare filename, so the
        images must be siblings of the .md or every image link breaks.
        """
        return self.rel.parent / self.stem


def resolve_input(input_path: Path) -> tuple[Path, str]:
    """Return (root_dir_to_walk, run_name).

    A zip is extracted once into work/extracted/<name>/ and reused on later runs.
    run_name is what outputs/<run_name>/ gets called.
    """
    input_path = input_path.expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"input not found: {input_path}")

    if input_path.is_file() and input_path.suffix.lower() == ".zip":
        name = input_path.stem
        dest = WORK / "extracted" / name
        if dest.exists() and any(dest.rglob("*")):
            return dest, name
        dest.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(input_path) as zf:
            # guard against zip-slip: never let a member escape dest
            for member in zf.infolist():
                target = (dest / member.filename).resolve()
                if not str(target).startswith(str(dest.resolve())):
                    raise RuntimeError(f"unsafe path in zip: {member.filename}")
            zf.extractall(dest)
        # a zip that contains a single top-level folder is the common case; descend into it
        entries = [p for p in dest.iterdir() if not p.name.startswith("__MACOSX")]
        if len(entries) == 1 and entries[0].is_dir():
            return entries[0], name
        return dest, name

    if input_path.is_file():
        return input_path.parent, input_path.parent.name

    return input_path, input_path.name


def find_docs(root: Path, include_images: bool = False) -> list[Doc]:
    """Every supported document under root, recursively, sorted for a stable run order."""
    wanted = set(PDF_SUFFIXES) | (IMAGE_SUFFIXES if include_images else set())
    docs = [
        Doc(path=p, rel=p.relative_to(root), root=root)
        for p in sorted(root.rglob("*"))
        if p.is_file() and p.suffix.lower() in wanted and not p.name.startswith(".")
    ]
    return docs


def clear_extracted(name: str) -> None:
    shutil.rmtree(WORK / "extracted" / name, ignore_errors=True)
