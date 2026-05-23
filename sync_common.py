#!/usr/bin/env python3
"""Sync ic_common source and dependencies to sibling ic_* projects.

Scans ../ic_* directories. For each that contains a common/ subdirectory:
  1. Copies all .py files and MODULE_CONTEXT.md from src/common/ -> <project>/common/
  2. Overwrites <project>/common/pyproject.toml with ic_common's pyproject.toml
  3. Adds any ic_common dependencies missing from <project>/pyproject.toml
"""

import re
import shutil
import tomllib
from pathlib import Path

_SYNC_EXTENSIONS = {".py", ".md"}
_SKIP_DIRS = {"__pycache__"}


def pkg_name(dep: str) -> str:
    """Normalize a dep specifier to a comparable package name."""
    return re.split(r"[>=<!;\[ ]", dep.strip())[0].strip().lower().replace("-", "_").replace(".", "_")


def sync_files(src_dir: Path, dst_dir: Path) -> int:
    """Copy source files to dst_dir, skipping unchanged ones. Returns number updated."""
    updated = 0
    for src in sorted(src_dir.iterdir()):
        if src.name in _SKIP_DIRS or src.is_dir():
            continue
        if src.suffix not in _SYNC_EXTENSIONS:
            continue
        dst = dst_dir / src.name
        if not dst.exists() or src.read_bytes() != dst.read_bytes():
            shutil.copy2(src, dst)
            print(f"    updated: {src.name}")
            updated += 1
    return updated


def sync_deps(ic_common_toml: Path, target_toml: Path) -> int:
    """Append any ic_common deps missing from target pyproject.toml. Returns count added."""
    with open(ic_common_toml, "rb") as f:
        src_data = tomllib.load(f)
    with open(target_toml, "rb") as f:
        dst_data = tomllib.load(f)

    src_deps = src_data.get("project", {}).get("dependencies", [])
    dst_deps = dst_data.get("project", {}).get("dependencies", [])
    dst_names = {pkg_name(d) for d in dst_deps}

    missing = [d for d in src_deps if pkg_name(d) not in dst_names]
    if not missing:
        return 0

    # Insert before the closing ] of the dependencies array
    lines = target_toml.read_text().splitlines(keepends=True)
    in_deps = False
    insert_at = None
    for i, line in enumerate(lines):
        if re.match(r"\s*dependencies\s*=\s*\[", line):
            in_deps = True
        if in_deps and re.match(r"\s*\]", line):
            insert_at = i
            break

    if insert_at is None:
        print(f"    WARNING: could not locate dependencies list in {target_toml.name}")
        return 0

    lines[insert_at:insert_at] = [f'    "{dep}",\n' for dep in missing]
    target_toml.write_text("".join(lines))

    for dep in missing:
        print(f"    added dep: {dep}")
    return len(missing)


def sync_project(ic_common: Path, target: Path) -> None:
    src_common = ic_common / "src" / "common"
    dst_common = target / "common"
    ic_toml = ic_common / "pyproject.toml"

    print(f"\n── {target.name}/")

    # 1. Sync source files
    n_files = sync_files(src_common, dst_common)
    if n_files == 0:
        print("    files: up to date")

    # 2. Overwrite common/pyproject.toml (kept as a mirror for reference)
    shutil.copy2(ic_toml, dst_common / "pyproject.toml")

    # 3. Merge missing deps into the project's main pyproject.toml
    target_toml = target / "pyproject.toml"
    if not target_toml.exists():
        print(f"    WARNING: {target.name}/pyproject.toml not found, skipping dep sync")
        return
    n_deps = sync_deps(ic_toml, target_toml)
    if n_deps == 0:
        print("    deps: up to date")


def main() -> None:
    ic_common = Path(__file__).parent.resolve()
    parent = ic_common.parent

    targets = sorted([
        d for d in parent.iterdir()
        if d.is_dir()
        and d.name.startswith("ic_")
        and d.name != ic_common.name
        and (d / "common").is_dir()
    ])

    if not targets:
        print("No sibling ic_* projects with a common/ directory found.")
        return

    print(f"Syncing to: {', '.join(t.name for t in targets)}")

    for target in targets:
        sync_project(ic_common, target)

    print("\nDone. Run `uv sync` in each updated project to install new dependencies.")


if __name__ == "__main__":
    main()
