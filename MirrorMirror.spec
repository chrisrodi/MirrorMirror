# -*- mode: python ; coding: utf-8 -*-
# Build:  .venv/bin/python -m PyInstaller --noconfirm MirrorMirror.spec
# Release (build + sign + DMG):  ./build_release.sh   (see --help)
import re
from pathlib import Path

_SRC = Path("mirror_mirror.py").read_text(encoding="utf-8")
APP_VERSION = re.search(r'^APP_VERSION\s*=\s*"([^"]+)"', _SRC, re.M).group(1)

a = Analysis(
    ['mirror_mirror.py'],
    pathex=[],
    binaries=[],
    datas=[],
    # pyobjc frameworks are loaded dynamically; make sure they are collected.
    hiddenimports=['objc', 'AppKit', 'Foundation', 'Vision', 'pypdf', 'docx', 'pptx'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    # Tesseract is an optional fallback; keep the bundle lean.
    excludes=['pytesseract'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='MirrorMirror',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,          # UPX breaks macOS code signing
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,   # builds for the arch of the running Python (arm64 here)
    codesign_identity=None,       # signing is done by build_release.sh
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='MirrorMirror',
)
app = BUNDLE(
    coll,
    name='MirrorMirror.app',
    icon='assets/icon.icns',
    bundle_identifier='com.christian.mirrormirror',
    version=APP_VERSION,
    info_plist={
        'CFBundleName': 'Mirror Mirror',
        'CFBundleDisplayName': 'Mirror Mirror',
        'CFBundleShortVersionString': APP_VERSION,
        'CFBundleVersion': APP_VERSION,
        'LSMinimumSystemVersion': '12.0',
        'LSApplicationCategoryType': 'public.app-category.productivity',
        'NSHighResolutionCapable': True,
        'NSHumanReadableCopyright': 'Copyright © 2026 Christian Rodriguez',
        # Screen capture has no purpose-string key; macOS prompts for
        # Screen Recording on first capture and the user enables it in
        # System Settings → Privacy & Security → Screen Recording.
    },
)
