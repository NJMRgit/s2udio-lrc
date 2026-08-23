#!/usr/bin/env bash
# Batch .lrc generation for the whole music library.
#   ./run_library.sh            full pass, simple format (word per line)
#   ./run_library.sh --enhanced full pass, enhanced format (line-based, inline
#                              <mm:ss.xx> word timestamps — s2udio karaoke mode)
#   ./run_library.sh --resume [--enhanced]  continue an interrupted run
#   ./run_library.sh --dry-run [--enhanced] preview
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIB="${LRCGEN_LIB:-/mnt/20TBHDD/Media/Music}"

ENHANCED=0
RESUME=0
DRY=0
EXTRA=()
for arg in "$@"; do
  case "$arg" in
    --enhanced) ENHANCED=1 ;;
    --resume)   RESUME=1 ;;
    --dry-run)  DRY=1 ;;
    *) EXTRA+=("$arg") ;;   # forwarded to lrcgen (--demucs, --model large-v3, ...)
  esac
done

LOG="$DIR/library.log"
[ "$ENHANCED" = 1 ] && LOG="$DIR/library-enhanced.log"

ARGS=()
[ "$ENHANCED" = 1 ] && ARGS+=(--format enhanced)
[ "$DRY" = 1 ] && ARGS+=(--dry-run)
if [ "$RESUME" = 1 ] && [ -s "$LOG" ]; then
  ARGS+=(--skip-from-log "$LOG")
  echo "resume mode: skipping tracks already logged OK/SKIP in $LOG"
fi

exec "$DIR/lrcgen" "$LIB" --log-file "$LOG" "${ARGS[@]}" ${EXTRA[@]+"${EXTRA[@]}"}
