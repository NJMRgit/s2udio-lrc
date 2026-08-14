#!/usr/bin/env python3
"""Apply the lyric-gap timing correction to existing enhanced .lrc files.

Whisper word timestamps run early after pauses: ~0.45s for the first word
after a gap, 0.15s for the second, 0.05s for the third (measured against
professionally-timed lyrics). This rewrites enhanced .lrc files in place,
delaying those words so karaoke highlighting matches the sung audio.

Rules (identical to lrcgen's built-in apply_gap_shift):
  - gap = word.start - previous_word.start > 2.0s  -> phrase reset
  - shifts: +0.45s (1st), +0.15s (2nd), +0.05s (3rd), then 0
  - monotonic word order preserved
  - line time re-anchored to the (shifted) first word
  - files already stamped `# lrcgen-gap-align:v1` are skipped (idempotent)
  - files without inline <mm:ss.xx> markers (plain/simple format) are skipped
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

STAMP = "# lrcgen-gap-align:v1"
GAP_THRESHOLD_S = 2.0
GAP_SHIFTS = (0.45, 0.15, 0.05)
LIB = __import__("os").environ.get("LRCGEN_LIB", "/mnt/20TBHDD/Media/Music")

LINE_RE = re.compile(r"^\[(\d+):(\d+\.\d+)\](.*)$")
WORD_RE = re.compile(r"<(\d+):(\d+\.\d+)>([^<]*)")


def parse_lyric_line(raw: str):
    """Return (line_time, [(word_time, word_text)]) or None if not a lyric line."""
    raw = raw.strip()
    m = LINE_RE.match(raw)
    if not m:
        return None
    mm, ss, body = m.groups()
    line_t = int(mm) * 60 + float(ss)
    words = []
    for wm in WORD_RE.finditer(body):
        wmm, wss, wtxt = wm.groups()
        t = int(wmm) * 60 + float(wss)
        words.append([t, wtxt.strip()])
    if not words:
        return None
    return (line_t, words)


def compute_shifts(times):
    """per-word shift seconds from the raw start-to-start gaps."""
    n = len(times)
    pos = [0] * n
    prev = None
    for i, t in enumerate(times):
        if prev is None or t - prev > GAP_THRESHOLD_S:
            pos[i] = 1
        else:
            pos[i] = min(pos[i - 1] + 1, len(GAP_SHIFTS) + 1)
        prev = t
    return [GAP_SHIFTS[p - 1] if 0 < p <= len(GAP_SHIFTS) else 0.0 for p in pos]


def fmt(t: float) -> str:
    m = int(t // 60)
    s = t - m * 60
    return f"{m:02d}:{s:05.2f}"


def process_file(path: Path) -> str:
    text = path.read_text(encoding="utf-8", errors="replace")
    if STAMP in text:
        return "skip-stamped"
    if "<" not in text:
        return "skip-plain"
    lines = text.splitlines()

    # header = everything before the first timestamp line
    header_end = 0
    for i, ln in enumerate(lines):
        if parse_lyric_line(ln) is not None:
            header_end = i
            break
    header = lines[:header_end]
    lyric_lines = lines[header_end:]

    parsed = [parse_lyric_line(ln) for ln in lyric_lines]
    if not any(p is not None for p in parsed):
        return "skip-nolyrics"

    # flatten to word stream, keep line structure
    line_refs = []          # (line_time, [word_idx...])
    times, texts = [], []
    for ln in lyric_lines:
        p = parse_lyric_line(ln)
        if p is None:
            continue
        line_t, words = p
        idxs = []
        for t, w in words:
            idxs.append(len(times))
            times.append(t)
            texts.append(w)
        line_refs.append((line_t, idxs))

    shifts = compute_shifts(times)

    # apply + monotonic clamp (even with no shifts, stamp so later runs skip it)
    new_times = [t + d for t, d in zip(times, shifts)]
    for i in range(1, len(new_times)):
        if new_times[i] < new_times[i - 1]:
            new_times[i] = new_times[i - 1]
    # end of a shifted word must not swallow the next word's start
    for i in range(len(new_times) - 1):
        if new_times[i + 1] - new_times[i] < 0.02:
            new_times[i + 1] = new_times[i] + 0.02

    # rebuild lyric lines
    out_lines = []
    for line_t, idxs in line_refs:
        first = new_times[idxs[0]]
        parts = [f"[{fmt(first)}]"]
        for idx in idxs:
            parts.append(f"<{fmt(new_times[idx])}>{texts[idx]} ")
        out_lines.append("".join(parts).rstrip())

    new_text = "\n".join(header + [STAMP] + out_lines) + "\n"
    if not any(shifts):
        path.write_text(new_text, encoding="utf-8")
        return "fixed (0 words shifted)"
    tmp = path.with_suffix(".lrc.tmp")
    tmp.write_text(new_text, encoding="utf-8")
    tmp.replace(path)
    n_shifted = sum(1 for d in shifts if d > 0)
    return f"fixed ({n_shifted} words shifted)"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="*", type=Path, default=[Path(LIB)],
                    help=".lrc files or directories (default: music library)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    files = []
    for p in args.paths:
        if p.is_dir():
            files.extend(f for f in p.rglob("*.lrc") if f.is_file())
        elif p.is_file() and p.suffix.lower() == ".lrc":
            files.append(p)
    files = sorted(set(files))

    stats = {"fixed": 0, "skip-stamped": 0, "skip-plain": 0,
             "skip-nolyrics": 0, "skip-noshift": 0, "error": 0}
    for f in files:
        try:
            r = process_file(f)
        except Exception as e:
            r = "error"
            print(f"ERROR {f}: {e}", file=sys.stderr)
        key = r.split()[0] if r.startswith("fixed") else r
        stats[key if key in stats else "error"] += 1
        if args.dry_run:
            print(f"  {r:16s} {f}")
        elif r.startswith("fixed"):
            print(f"  {r:16s} {f}")

    print(f"\n{sum(stats.values())} files: " +
          ", ".join(f"{k}={v}" for k, v in stats.items() if v))
    return 0


if __name__ == "__main__":
    sys.exit(main())
