#!/usr/bin/env bash
# Nightly max-accuracy .lrc rebuild for the whole music library.
# Runs only inside the 04:00-10:00 window, resumes where it left off,
# repeats every night until every track is done, then exits.
set -u
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIB="${LRCGEN_LIB:-/mnt/20TBHDD/Media/Music}"
LOG="$DIR/library-enhanced.log"

next_4am() {
  local now t4
  now=$(date +%s)
  t4=$(date -d "today 04:00" +%s)
  [ "$now" -lt "$t4" ] && { echo "$t4"; return; }
  date -d "tomorrow 04:00" +%s
}

secs_to_10am() {
  local now t10
  now=$(date +%s)
  t10=$(date -d "today 10:00" +%s)
  echo $(( t10 > now ? t10 - now : 0 ))
}

remaining() {
  "$DIR/lrcgen" "$LIB" --format enhanced --dry-run --skip-from-log "$LOG" 2>/dev/null \
    | sed -n 's/^found [0-9]* audio file(s); \([0-9]*\) to (re)generate.*/\1/p'
}

while :; do
  left=$(remaining)
  if [ "${left:-1}" = "0" ]; then
    echo "$(date '+%F %T') all tracks rebuilt with max-accuracy pipeline; exiting"
    break
  fi
  echo "$(date '+%F %T') $left track(s) remaining; next window starts at $(date -d @$(next_4am) '+%F %T')"
  sleep $(( $(next_4am) - $(date +%s) ))

  # clean any demucs temp dirs a hard stop may have left behind
  find "$LIB" -type d -name lrcgen-demucs -prune -exec rm -rf {} + 2>/dev/null

  dur=$(secs_to_10am)
  echo "$(date '+%F %T') window open: rebuilding for ${dur}s (resume from log)"
  timeout --signal=TERM "$dur" \
    "$DIR/run_library.sh" --enhanced --resume \
      --demucs --model large-v3 --compute-type float16
  echo "$(date '+%F %T') window closed (rc=$?)"
done