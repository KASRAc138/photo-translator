#!/usr/bin/env python3
"""Build the portable folder and the .exe from this source tree.

    python build_portable.py              # portable folder + onefile exe
    python build_portable.py --onedir     # portable folder only (starts faster)
    python build_portable.py --onefile    # single exe only
    python build_portable.py --check      # report what would be bundled, build nothing

Two artefacts, one source tree:

* ``dist/PhotoTranslator/`` -- a portable folder with the interpreter and every
  dependency vendored. Copy it to a USB stick and it runs.
* ``dist/PhotoTranslator.exe`` -- one file. Unpacks to a temp dir on each
  launch, so it starts a few seconds slower, but it is a single thing to hand
  someone.

Either way, on first run the app creates ``input/`` and ``output/`` **next to
itself** and prints where they are.

Three things PyInstaller gets wrong here without help, each of which produces a
build that runs on this machine and fails on the user's:

1. **RapidOCR's ONNX models** are loaded by path at runtime, so the dependency
   analyser never sees them and the exe ships without the ~15 MB it needs.
2. **Pillow's Raqm/HarfBuzz DLLs** are not Python imports. Losing them silently
   downgrades Persian and Arabic to the reshaper fallback -- no error, just
   worse text -- which is why :func:`check_raqm` fails the build loudly instead.
3. **CTranslate2's native libraries** back Argos Translate and are likewise
   invisible to static analysis.
4. **The web UI's HTML** lives in ``pt/static/`` and is read from disk at
   request time. Miss it and the exe runs, binds its port, and serves a 404
   for its own front page.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NAME = "PhotoTranslator"
ENTRY = ROOT / "PhotoTranslator.py"

# Packages whose data files must be collected wholesale. The comment on each
# is the failure you get when it is missing.
COLLECT_ALL = [
    "rapidocr_onnxruntime",  # models are loaded by path -> "model file not found"
    "argostranslate",        # package metadata -> "no translation available"
    "ctranslate2",           # native .dll/.so -> ImportError at first translate
    "arabic_reshaper",       # ships a config .ini -> crash on RTL fallback
]

HIDDEN_IMPORTS = [
    "onnxruntime",
    "PIL._imaging",
    "PIL._imagingft",   # FreeType; without it every draw call fails
    "sentencepiece",
    "minisbd",          # the sentence splitter that replaces stanza
]

# Big, optional, and not on the default path.
#
# Measured on a Linux build of this tree: excluding these took the portable
# folder from 887 MB to 679 MB, almost all of it spacy/blis/thinc. Windows
# builds come out substantially smaller because the opencv and ctranslate2
# native libraries are far smaller there -- expect roughly 250-400 MB for the
# folder and less for the compressed onefile exe. What remains is genuinely
# required: opencv (RapidOCR depends on it) and ctranslate2 (Argos runs on it).
EXCLUDES = [
    # The Unlimited-OCR tier. Opt-in, installed separately on a GPU machine.
    "torch", "transformers", "accelerate",

    # Argos pulls these in for sentence splitting. stanza needs torch, and
    # spacy costs ~150 MB with blis/thinc; pt.translate installs a stanza stub
    # so Argos falls back to MiniSBD, which is pure Python and adequate here.
    "stanza", "spacy", "blis", "thinc", "cupy", "spacy_legacy", "spacy_loggers",
    "weasel", "srsly", "catalogue", "confection", "preshed", "cymem", "murmurhash",

    # opencv-python and opencv-python-headless install the same module under
    # two distributions. Bundling both duplicates ~100 MB of native libraries.
    "opencv-python",

    # Never on any code path this app takes.
    "matplotlib", "scipy", "pandas", "IPython", "notebook", "tkinter", "pytest",
    "lxml", "cryptography", "setuptools", "pip", "wheel", "fontTools",
]


# ---------------------------------------------------------------------------
# Pre-flight
# ---------------------------------------------------------------------------


def have(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def check_deps() -> list[str]:
    required = {
        "PIL": "pillow",
        "numpy": "numpy",
        "cv2": "opencv-python-headless",
        "rapidocr_onnxruntime": "rapidocr-onnxruntime",
        "argostranslate": "argostranslate",
    }
    missing = [pkg for mod, pkg in required.items() if not have(mod)]
    optional = {"arabic_reshaper": "arabic-reshaper", "bidi": "python-bidi"}
    for mod, pkg in optional.items():
        if not have(mod):
            print(f"  note: {pkg} is not installed (RTL fallback unavailable)")
    return missing


def check_raqm() -> bool:
    """Warn loudly if Pillow cannot shape complex scripts.

    A build without Raqm still produces correct Latin output, so this would
    otherwise go unnoticed until a Persian user reports "the letters are
    separate" -- weeks later, with no error to point at.
    """
    try:
        from PIL import features
        ok = bool(features.check("raqm"))
    except Exception:
        ok = False
    if not ok:
        print(
            "  WARNING: Pillow has no Raqm/HarfBuzz support.\n"
            "           Persian and Arabic will use the arabic-reshaper fallback,\n"
            "           which handles fewer cases. Fix with:\n"
            "               pip install --upgrade --force-reinstall pillow"
        )
    return ok


def bundled_fonts() -> list[Path]:
    folder = ROOT / "fonts"
    return sorted(folder.glob("*.ttf")) + sorted(folder.glob("*.otf")) if folder.is_dir() else []


def check_fonts() -> None:
    """A build with no Arabic-capable font cannot produce Persian output.

    On Windows the app will find arial.ttf at runtime, so this is a warning,
    not an error -- but a portable folder handed to a stranger should carry
    its own font rather than hoping.
    """
    fonts = bundled_fonts()
    if not fonts:
        print("  note: fonts/ is empty. The app will use system fonts at runtime.")
        return
    sys.path.insert(0, str(ROOT))
    try:
        from pt.textshape import coverage
        best = max((coverage(str(f), "fa"), f.name) for f in fonts)
        if best[0] < 0.9:
            print(f"  WARNING: no bundled font covers Persian (best: {best[1]} at "
                  f"{best[0]:.0%}). Add Vazirmatn or Noto Naskh Arabic to fonts/.")
        else:
            print(f"  fonts: {len(fonts)} bundled, Persian covered by {best[1]}")
    except Exception as exc:
        print(f"  note: could not verify font coverage ({exc})")


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


def pyinstaller_args(onefile: bool) -> list[str]:
    sep = ";" if os.name == "nt" else ":"
    args = [
        sys.executable, "-m", "PyInstaller",
        "--noconfirm", "--clean",
        "--name", NAME,
        "--console",
        "--onefile" if onefile else "--onedir",
        "--distpath", str(ROOT / "dist"),
        "--workpath", str(ROOT / "build"),
        "--specpath", str(ROOT / "build"),
    ]
    for pkg in COLLECT_ALL:
        if have(pkg):
            args += ["--collect-all", pkg]
    for mod in HIDDEN_IMPORTS:
        if have(mod.split(".")[0]):
            args += ["--hidden-import", mod]
    for mod in EXCLUDES:
        args += ["--exclude-module", mod]
    if bundled_fonts():
        args += ["--add-data", f"{ROOT / 'fonts'}{sep}fonts"]
    # The web UI is a data file, not an import. Without this the exe starts,
    # binds its port, and serves a 404 for the page itself.
    args += ["--add-data", f"{ROOT / 'pt' / 'static'}{sep}pt/static"]
    icon = ROOT / "assets" / "icon.ico"
    if icon.is_file():
        args += ["--icon", str(icon)]
    args.append(str(ENTRY))
    return args


def run_pyinstaller(onefile: bool) -> Path:
    args = pyinstaller_args(onefile)
    print(f"\n  pyinstaller {'--onefile' if onefile else '--onedir'} ...")
    result = subprocess.run(args, cwd=ROOT)
    if result.returncode != 0:
        raise SystemExit(f"PyInstaller failed with exit code {result.returncode}")
    exe = NAME + (".exe" if os.name == "nt" else "")
    return ROOT / "dist" / exe if onefile else ROOT / "dist" / NAME / exe


def stage_portable(target_dir: Path) -> None:
    """Add the human-facing bits next to the exe: launcher, README, folders."""
    target_dir.mkdir(parents=True, exist_ok=True)
    (target_dir / "input").mkdir(exist_ok=True)
    (target_dir / "output").mkdir(exist_ok=True)

    for name in ("README.md", "LICENSE"):
        src = ROOT / name
        if src.is_file():
            shutil.copy2(src, target_dir / name)

    exe_name = NAME + (".exe" if os.name == "nt" else "")
    (target_dir / "Translate photos.bat").write_text(
        "@echo off\r\n"
        "cd /d \"%~dp0\"\r\n"
        f"\"{exe_name}\"\r\n",
        encoding="utf-8",
    )
    (target_dir / "Translate to Persian.bat").write_text(
        "@echo off\r\n"
        "cd /d \"%~dp0\"\r\n"
        "set PT_TO=fa\r\n"
        f"\"{exe_name}\"\r\n",
        encoding="utf-8",
    )


def zip_folder(folder: Path, archive: Path) -> Path:
    archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in sorted(folder.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(folder.parent))
    return archive


def human(size: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.0f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build Photo Translator.")
    parser.add_argument("--onefile", action="store_true", help="single exe only")
    parser.add_argument("--onedir", action="store_true", help="portable folder only")
    parser.add_argument("--check", action="store_true", help="pre-flight only")
    parser.add_argument("--no-zip", action="store_true")
    args = parser.parse_args(argv)

    print(f"Building {NAME}\n  python {sys.version.split()[0]} on {sys.platform}")

    missing = check_deps()
    if missing:
        print("\n  Missing required packages:\n    pip install " + " ".join(missing))
        return 1
    check_raqm()
    check_fonts()

    if not have("PyInstaller"):
        print("\n  PyInstaller is not installed:\n    pip install pyinstaller")
        return 1
    if args.check:
        print("\n  Pre-flight only; nothing built.")
        return 0

    if os.name != "nt":
        print("\n  NOTE: building on a non-Windows host produces a non-Windows\n"
              "        binary. Run this on Windows to get PhotoTranslator.exe.")

    build_dir = not args.onefile or args.onedir
    build_file = not args.onedir or args.onefile
    if args.onefile and not args.onedir:
        build_dir = False
    if args.onedir and not args.onefile:
        build_file = False

    produced: list[Path] = []
    if build_dir:
        exe = run_pyinstaller(onefile=False)
        folder = exe.parent
        stage_portable(folder)
        produced.append(folder)
        if not args.no_zip:
            archive = zip_folder(folder, ROOT / "dist" / f"{NAME}-portable.zip")
            produced.append(archive)
    if build_file:
        exe = run_pyinstaller(onefile=True)
        stage_portable(exe.parent)
        produced.append(exe)

    print("\n  Built:")
    for path in produced:
        if path.is_file():
            print(f"    {path}  ({human(path.stat().st_size)})")
        else:
            total = sum(p.stat().st_size for p in path.rglob('*') if p.is_file())
            print(f"    {path}{os.sep}  ({human(total)})")

    print(
        "\n  First run on a new machine downloads the language pack (~100 MB, once).\n"
        "  To pre-seed it so the shipped copy works offline:\n"
        f"      {NAME} --setup --to fa\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
