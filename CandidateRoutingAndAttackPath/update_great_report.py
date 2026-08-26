from __future__ import annotations

import shutil
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT = Path(
    "/Users/ashentide/Library/Mobile Documents/com~apple~CloudDocs/Obsidian/Research/"
    "3 - Papers/Большие отчеты/Великий отчёт 1.md"
)
ACADEMIC_REPORT = REPORT.with_name("Великий отчёт 1 — академическая версия.md")
ASSET_DIR = Path(
    "/Users/ashentide/Library/Mobile Documents/com~apple~CloudDocs/Obsidian/Research/"
    "0 - Assets/Images"
)
SOURCE_HEADING = "# Список использованных источников"
NEW_SECTION_HEADING = "## Причинный анализ того, как патч подавляет детекцию"
ACADEMIC_SECTION_HEADING = "## Причинный анализ механизма подавления детекции"


def _insert_before_sources(body: str, fragment: str, *, existing_heading: str) -> str:
    if existing_heading in body:
        raise RuntimeError(f"Section already exists: {existing_heading}")
    if body.count(SOURCE_HEADING) != 1:
        raise RuntimeError(
            f"Expected one source heading, found {body.count(SOURCE_HEADING)}"
        )
    prefix, sources = body.split(SOURCE_HEADING, maxsplit=1)
    return (
        prefix.rstrip()
        + "\n\n"
        + fragment.strip()
        + "\n\n"
        + SOURCE_HEADING
        + sources
    )


def main() -> None:
    original_fragment = (
        REPO_ROOT / "CandidateRoutingAndAttackPath/report_appendix_original_style.md"
    ).read_text(encoding="utf-8")
    academic_fragment = (
        REPO_ROOT / "CandidateRoutingAndAttackPath/report_appendix_academic_style.md"
    ).read_text(encoding="utf-8")
    original_body = REPORT.read_text(encoding="utf-8")

    updated_original = _insert_before_sources(
        original_body,
        original_fragment,
        existing_heading=NEW_SECTION_HEADING,
    )
    academic_body = _insert_before_sources(
        original_body,
        academic_fragment,
        existing_heading=ACADEMIC_SECTION_HEADING,
    )

    assets = REPO_ROOT / "CandidateRoutingAndAttackPath/report_assets"
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    for source in sorted(assets.glob("functional-component-*.png")):
        target = ASSET_DIR / source.name
        if target.exists() and target.read_bytes() != source.read_bytes():
            raise RuntimeError(f"Refusing to overwrite different asset: {target}")
        if not target.exists():
            shutil.copy2(source, target)

    REPORT.write_text(updated_original, encoding="utf-8")
    ACADEMIC_REPORT.write_text(academic_body, encoding="utf-8")
    print(REPORT)
    print(ACADEMIC_REPORT)


if __name__ == "__main__":
    main()
