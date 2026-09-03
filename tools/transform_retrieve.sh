#!/bin/bash
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:$PATH"
cd "$(dirname "$0")/.." || exit 3
exec "$HOME/.local/bin/uv" run python tools/transform_retrieve.py
