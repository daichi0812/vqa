#!/bin/bash
# scripts/setup_data.sh
#
# vqa プロジェクトのデータをセットアップするスクリプト。
# 新しいマシンで `git clone` した後、最初に走らせる想定。
#
# 動作:
#   1. dev-assets の場所を自動検出 (Mac / Colab)
#   2. data.zip, model.pt, submission.* のシンボリックリンクをリポジトリ直下に作成
#   3. data.zip を data/ に展開（未展開時のみ）
#   4. manifest.yaml のハッシュと付き合わせて検証 (デフォルトは小ファイルのみ)
#
# 使い方:
#   bash scripts/setup_data.sh                # 通常のセットアップ
#   bash scripts/setup_data.sh --verify-all   # data.zip のハッシュも検証
#   bash scripts/setup_data.sh --no-verify    # 検証スキップ
#   bash scripts/setup_data.sh --no-extract   # data.zip 展開をスキップ

set -euo pipefail

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROJECT_REL="events/kaggle/vqa"  # dev-hub および dev-assets 内での共通パス
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# 同期可能性のある dev-assets ベースパス候補（前から順に試す）
ASSET_CANDIDATES=(
  "$HOME/Library/CloudStorage/GoogleDrive-daichi8120@gmail.com/My Drive/dev-assets"  # Mac
  "/content/drive/MyDrive/dev-assets"                                                # Colab
  "$HOME/My Drive/dev-assets"                                                        # Drive desktop fallback
)

# repo相対パス | dev-assets相対パス
LINK_SPECS=(
  "data.zip|raw/data.zip"
  "models/model.pt|models/model.pt"
  "outputs/submission.zip|outputs/submission-2025-01-05/submission.zip"
  "outputs/submission.npy|outputs/submission-2025-01-05/submission.npy"
)

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
VERIFY_MODE="small"   # small | all | none
DO_EXTRACT=1
for arg in "$@"; do
  case "$arg" in
    --verify-all) VERIFY_MODE="all" ;;
    --no-verify)  VERIFY_MODE="none" ;;
    --no-extract) DO_EXTRACT=0 ;;
    -h|--help)
      sed -n '2,18p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'
      exit 0 ;;
    *)
      echo "ERROR: unknown option: $arg" >&2
      exit 1 ;;
  esac
done

# ---------------------------------------------------------------------------
# Phase 1: dev-assets を見つける
# ---------------------------------------------------------------------------
ASSETS=""
for base in "${ASSET_CANDIDATES[@]}"; do
  candidate="$base/$PROJECT_REL"
  if [ -d "$candidate" ]; then
    ASSETS="$candidate"
    break
  fi
done

if [ -z "$ASSETS" ]; then
  cat >&2 <<EOM
ERROR: dev-assets/$PROJECT_REL が見つかりませんでした。

検出を試みた場所:
$(for b in "${ASSET_CANDIDATES[@]}"; do echo "  - $b/$PROJECT_REL"; done)

対処:
  Mac:   Google Drive デスクトップアプリでサインインし、dev-assets を同期する
  Colab: 先頭セルで以下を実行
           from google.colab import drive
           drive.mount('/content/drive')
  その他: rclone か gdown で dev-assets/$PROJECT_REL を取得してから再実行
EOM
  exit 1
fi

echo "==> dev-assets found: $ASSETS"
echo ""

# ---------------------------------------------------------------------------
# Phase 2: シンボリックリンク作成
# ---------------------------------------------------------------------------
echo "==> Creating symlinks in $REPO_DIR"
for spec in "${LINK_SPECS[@]}"; do
  IFS='|' read -r repo_rel asset_rel <<< "$spec"
  src="$ASSETS/$asset_rel"
  dst="$REPO_DIR/$repo_rel"
  if [ ! -e "$src" ]; then
    echo "  [skip] $repo_rel (source missing: $src)"
    continue
  fi
  mkdir -p "$(dirname "$dst")"
  ln -sfn "$src" "$dst"
  echo "  [ok]   $repo_rel -> $src"
done
echo ""

# ---------------------------------------------------------------------------
# Phase 3: data.zip を data/ に展開
# ---------------------------------------------------------------------------
EXTRACT_MARKER="$REPO_DIR/data/.extracted"
DATA_ZIP="$REPO_DIR/data.zip"

if [ "$DO_EXTRACT" -eq 0 ]; then
  echo "==> --no-extract: skipping data.zip extraction"
elif [ -f "$EXTRACT_MARKER" ]; then
  echo "==> data/ already extracted (.extracted marker present)"
elif [ ! -e "$DATA_ZIP" ]; then
  echo "==> data.zip not available, skipping extraction"
else
  if ! command -v unzip >/dev/null; then
    echo "ERROR: unzip not found. Install it (e.g. apt install unzip / brew install unzip)" >&2
    exit 1
  fi
  echo "==> Extracting data.zip to $REPO_DIR (zip contains data/ at top level)"
  unzip -oq "$DATA_ZIP" -d "$REPO_DIR"
  touch "$EXTRACT_MARKER"
  echo "    Extracted."
fi
echo ""

# ---------------------------------------------------------------------------
# Phase 4: ハッシュ検証
# ---------------------------------------------------------------------------
MANIFEST="$ASSETS/manifest.yaml"

if [ "$VERIFY_MODE" = "none" ]; then
  echo "==> --no-verify: skipping hash verification"
elif [ ! -f "$MANIFEST" ]; then
  echo "WARNING: manifest.yaml not found at $MANIFEST, skipping verification"
elif ! command -v python3 >/dev/null; then
  echo "WARNING: python3 not available, skipping verification"
else
  echo "==> Verifying hashes (mode: $VERIFY_MODE)"
  python3 - "$MANIFEST" "$REPO_DIR" "$VERIFY_MODE" <<'PYEOF'
import sys, os, hashlib

manifest_path, repo_dir, mode = sys.argv[1], sys.argv[2], sys.argv[3]

try:
    import yaml
except ImportError:
    print("  PyYAML not installed; install with: pip install pyyaml", file=sys.stderr)
    print("  Skipping hash verification.")
    sys.exit(0)

with open(manifest_path) as f:
    m = yaml.safe_load(f)

# (path in repo, expected sha256, is_large)
checks = []
data = m.get("data", {}) or {}
if "raw_zip" in data:
    checks.append(("data.zip", data["raw_zip"]["sha256"], True))

models = m.get("models", {}) or {}
if "baseline" in models:
    checks.append(("models/model.pt", models["baseline"]["sha256"], False))

outputs = m.get("outputs", {}) or {}
for sub in outputs.values():
    for f in sub.get("files", []) or []:
        name = os.path.basename(f["path"])
        if name.startswith("submission."):
            checks.append((f"outputs/{name}", f["sha256"], False))

ok = fail = skipped = 0
for fname, expected, is_large in checks:
    if mode == "small" and is_large:
        print(f"  [skip] {fname} (large file, use --verify-all to include)")
        skipped += 1
        continue
    path = os.path.join(repo_dir, fname)
    if not os.path.exists(path):
        print(f"  [skip] {fname} (not found in repo)")
        skipped += 1
        continue

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    actual = h.hexdigest()

    if actual == expected:
        print(f"  [ok]   {fname}  {actual[:12]}...")
        ok += 1
    else:
        print(f"  [FAIL] {fname}")
        print(f"           expected: {expected}")
        print(f"           actual:   {actual}")
        fail += 1

print(f"  Summary: ok={ok}  fail={fail}  skipped={skipped}")
sys.exit(1 if fail else 0)
PYEOF
fi
echo ""

echo "==> Setup complete"
