#!/usr/bin/env python3
"""
Download TTF files for every font in our databases using git sparse-checkout
from the Google Fonts GitHub repo. No API keys, no rate limits.

Run once: python scripts/download_fonts.py
Fonts saved to: fonts/ttf/
"""
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

FONTS_DIR = Path(__file__).parent.parent / "fonts"
TTF_DIR   = FONTS_DIR / "ttf"
TTF_DIR.mkdir(parents=True, exist_ok=True)

KEEP_STYLES   = {"Bold", "ExtraBold", "Black", "SemiBold", "Regular"}
SKIP_PATTERNS = {"Italic", "italic"}

# Fonts whose Google Fonts directory name differs from the family name
DIR_OVERRIDES = {
    "Fredoka One":      "fredoka",
    "Old Standard TT":  "oldstandardtt",
    "PT Sans":          "ptsans",
    "PT Serif":         "ptserif",
    "Nanum Myeongjo":   "nanummyeongjo",
    "Slabo 27px":       "slabo27px",
}

# Fonts not on Google Fonts (skip gracefully)
NOT_ON_GOOGLE_FONTS = {"Impact", "Proxima Nova", "Helvetica Neue"}


def dir_name(family: str) -> str:
    if family in DIR_OVERRIDES:
        return DIR_OVERRIDES[family]
    return re.sub(r"[\s\-]", "", family).lower()


def load_all_font_names() -> set[str]:
    names: set[str] = set()
    for fname in ("tiktok.json", "capcut.json", "canva_popular.json"):
        path = FONTS_DIR / fname
        if not path.exists():
            continue
        for entry in json.loads(path.read_text()):
            for key in ("google_font", "google_font_equivalent", "fallback_google"):
                gf = entry.get(key)
                if gf and gf not in NOT_ON_GOOGLE_FONTS:
                    names.add(gf)
    return names


EXTRA = [
    "Nunito Sans", "Plus Jakarta Sans", "Figtree", "Black Ops One",
    "Bowlby One SC", "Passion One", "Teko", "Saira Condensed",
    "Rajdhani", "Audiowide", "Orbitron", "Syncopate",
    "Encode Sans Condensed", "Kaushan Script", "Shadows Into Light",
    "Patrick Hand", "Rock Salt", "Reenie Beanie", "Architects Daughter",
    "Handlee", "Covered By Your Grace", "Just Another Hand",
    "Michroma", "Share Tech Mono", "Sigmar One", "League Gothic", "League Spartan",
    "Indie Flower", "Permanent Marker", "Lobster", "Pacifico",
    "Dancing Script", "Oswald", "Raleway", "Montserrat", "Poppins",
    "Lato", "Inter", "Roboto", "Open Sans", "Nunito", "Playfair Display",
    "Merriweather", "Work Sans", "Rubik", "Josefin Sans",
]


def copy_ttfs(src_root: Path, dst_dir: Path) -> int:
    copied = 0
    for ttf in src_root.rglob("*.ttf"):
        stem = ttf.stem
        # Skip italics
        if any(p in stem for p in SKIP_PATTERNS):
            continue
        # Variable fonts → rename [wght] → Variable
        if "[" in stem:
            family = re.split(r"[\[\-]", stem)[0].rstrip()
            dest   = dst_dir / f"{family}-Variable.ttf"
        elif "-" in stem:
            parts  = stem.rsplit("-", 1)
            style  = parts[1]
            if style not in KEEP_STYLES:
                continue
            dest = dst_dir / ttf.name
        else:
            dest = dst_dir / f"{stem}-Regular.ttf"

        if not dest.exists():
            shutil.copy2(ttf, dest)
            copied += 1
    return copied


if __name__ == "__main__":
    all_fonts = sorted(load_all_font_names() | set(EXTRA))
    sparse_dirs = []
    for family in all_fonts:
        dname = dir_name(family)
        sparse_dirs += [f"ofl/{dname}", f"apache/{dname}", f"ufl/{dname}"]

    print(f"Cloning Google Fonts repo (sparse, no blobs)...")
    with tempfile.TemporaryDirectory() as tmpdir:
        repo = Path(tmpdir) / "fonts"
        subprocess.run(
            ["git", "clone", "--depth=1", "--filter=blob:none", "--no-checkout",
             "https://github.com/google/fonts.git", str(repo)],
            check=True, capture_output=True
        )
        subprocess.run(
            ["git", "sparse-checkout", "init", "--cone"],
            cwd=repo, check=True, capture_output=True
        )
        subprocess.run(
            ["git", "sparse-checkout", "set"] + sparse_dirs,
            cwd=repo, check=True, capture_output=True
        )
        subprocess.run(
            ["git", "checkout"],
            cwd=repo, check=True, capture_output=True
        )

        ttf_count_before = len(list(TTF_DIR.glob("*.ttf")))
        copied = copy_ttfs(repo, TTF_DIR)
        ttf_count_after  = len(list(TTF_DIR.glob("*.ttf")))

    print(f"Done. Copied {copied} new TTFs. Total in fonts/ttf/: {ttf_count_after}")

    # Report what's still missing
    have = {re.sub(r'-(Bold|ExtraBold|Black|SemiBold|Regular|Variable).*', '', p.stem).lower().replace(' ', '')
            for p in TTF_DIR.glob('*.ttf')}
    missing = [f for f in all_fonts if dir_name(f) not in have]
    if missing:
        print(f"Still missing ({len(missing)}): {', '.join(missing)}")
    else:
        print("All target fonts downloaded.")
