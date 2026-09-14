#!/usr/bin/env bash
# Build a distributable Mirror Mirror DMG.
#
#   ./build_release.sh                       # ad-hoc signed (free; users must "Open Anyway" once)
#   ./build_release.sh --identity "Developer ID Application: Name (TEAMID)"
#   ./build_release.sh --identity "..." --notarize notary-profile
#   ./build_release.sh --skip-build          # reuse the previous build in ~/Library/Caches/MirrorMirror/build
#
# Output: release/MirrorMirror-<version>.dmg and a .sha256 next to it.
# Work tree: ~/Library/Caches/MirrorMirror/build (see MM_BUILD_DIR below).
#
# For --notarize, create the keychain profile once with:
#   xcrun notarytool store-credentials notary-profile \
#       --apple-id you@example.com --team-id TEAMID --password <app-specific-password>
set -euo pipefail

cd "$(dirname "$0")"

IDENTITY=""
NOTARIZE_PROFILE=""
SKIP_BUILD=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --identity) IDENTITY="$2"; shift 2 ;;
    --notarize) NOTARIZE_PROFILE="$2"; shift 2 ;;
    --skip-build) SKIP_BUILD=1; shift ;;
    -h|--help) sed -n '2,15p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

PY=.venv/bin/python
# Everything is built and signed OUTSIDE the project folder. This project lives in
# an iCloud-synced folder (Desktop), and iCloud's file provider keeps re-tagging
# .app/.framework directories with Finder xattrs within seconds, which makes
# codesign refuse the bundle. ~/Library/Caches is not synced. Override with
# MM_BUILD_DIR if you want the work tree somewhere else.
WORK="${MM_BUILD_DIR:-$HOME/Library/Caches/MirrorMirror/build}"
APP=$WORK/dist/MirrorMirror.app
VERSION=$(sed -nE 's/^APP_VERSION *= *"([^"]+)".*/\1/p' mirror_mirror.py)
[[ -n "$VERSION" ]] || { echo "could not read APP_VERSION from mirror_mirror.py" >&2; exit 1; }
DMG_NAME=MirrorMirror-${VERSION}.dmg
DMG=$WORK/$DMG_NAME

echo "==> Mirror Mirror ${VERSION}"

if [[ $SKIP_BUILD -eq 0 ]]; then
  echo "==> Icon"
  [[ -f assets/icon.icns ]] || $PY assets/make_icon.py
  echo "==> PyInstaller"
  mkdir -p "$WORK"
  $PY -m PyInstaller --noconfirm --log-level WARN \
      --workpath "$WORK/build" --distpath "$WORK/dist" MirrorMirror.spec
fi
[[ -d "$APP" ]] || { echo "missing $APP" >&2; exit 1; }

echo "==> Stripping Finder metadata (breaks codesign otherwise)"
# xattrs, .DS_Store files and BSD flags such as "hidden" all count as
# "Finder information" and make codesign refuse the bundle.
xattr -cr "$APP"
chflags -R nohidden,nouchg "$APP"
find "$APP" -name .DS_Store -delete

if [[ -n "$IDENTITY" ]]; then
  echo "==> Signing with Developer ID (hardened runtime)"
  # Sign nested code first, then the bundle, so the seal covers everything.
  find "$APP/Contents" \( -name "*.dylib" -o -name "*.so" -o -name "*.framework" \) -print0 \
    | xargs -0 -I{} codesign --force --options runtime --timestamp \
        --entitlements entitlements.plist -s "$IDENTITY" "{}"
  codesign --force --options runtime --timestamp \
    --entitlements entitlements.plist -s "$IDENTITY" "$APP"
else
  echo "==> Ad-hoc signing (no Developer ID given)"
  codesign --force --deep -s - "$APP"
fi
codesign --verify --deep --strict "$APP"
echo "    signature OK"

echo "==> DMG"
STAGE=$(mktemp -d)
cp -R "$APP" "$STAGE/"
ln -s /Applications "$STAGE/Applications"
rm -f "$DMG"
hdiutil create -quiet -volname "Mirror Mirror" -srcfolder "$STAGE" -ov -format UDZO "$DMG"
rm -rf "$STAGE"

if [[ -n "$IDENTITY" ]]; then
  codesign --force --timestamp -s "$IDENTITY" "$DMG"
fi

if [[ -n "$NOTARIZE_PROFILE" ]]; then
  [[ -n "$IDENTITY" ]] || { echo "--notarize requires --identity" >&2; exit 1; }
  echo "==> Notarizing (this waits for Apple)"
  xcrun notarytool submit "$DMG" --keychain-profile "$NOTARIZE_PROFILE" --wait
  xcrun stapler staple "$DMG"
  spctl --assess --type open --context context:primary-signature -v "$DMG" || true
fi

mkdir -p release
cp -f "$DMG" "release/$DMG_NAME"
(cd release && shasum -a 256 "$DMG_NAME" | tee "$DMG_NAME.sha256")
echo "==> Done: release/$DMG_NAME"
if [[ -z "$IDENTITY" ]]; then
  cat <<'EOF'

NOTE: ad-hoc signed. Downloaders will see "Apple could not verify..." on first launch.
Tell them: open it once, then System Settings → Privacy & Security → "Open Anyway".
Add --identity/--notarize once you have a Developer ID certificate to remove that step.
EOF
fi
