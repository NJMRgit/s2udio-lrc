#!/usr/bin/env python3
"""
lrcgen — word-per-word .lrc sidecar generator for a local music library.

Engine: faster-whisper (CTranslate2 Whisper) running locally on CUDA.
Model:  Systran/faster-whisper-large-v3-turbo by default (downloads on first run
        into ~/.cache/huggingface). Use --model large-v3 for maximum accuracy.

Output formats
--------------
  --format simple    (default) one line per word:
                         [00:12.34]word
  --format enhanced  one line per lyric line, each word carries its own timestamp
                     (karaoke style, supported by mpv, foobar2000, Musixmatch):
                         [00:12.34]<00:12.34>word <00:12.62>word2

By default every audio file is transcribed and its .lrc (re)written — existing
.lrc files are NOT trusted. Use --skip-existing to leave them alone.
"""
from __future__ import annotations

import argparse
import os
import site
import sys
import time
from pathlib import Path

# --- CUDA 12 runtime bootstrap -------------------------------------------
# The system has CUDA 13 only; ctranslate2 needs CUDA 12 libs (libcublas.so.12,
# libcudnn.so.9, libcudart.so.12). Those ship as pip wheels (nvidia-*-cu12) in
# this venv's site-packages — expose them via LD_LIBRARY_PATH before ctranslate2
# is imported.
try:
    _nv = Path(site.getsitepackages()[0]) / "nvidia"
    _libdirs = [str(d) for d in _nv.glob("*/lib") if d.is_dir()] if _nv.is_dir() else []
    if _libdirs:
        _existing = os.environ.get("LD_LIBRARY_PATH", "")
        os.environ["LD_LIBRARY_PATH"] = ":".join(_libdirs + ([_existing] if _existing else []))
except Exception:
    pass
# -------------------------------------------------------------------------

AUDIO_EXT = {".mp3", ".flac", ".m4a", ".ogg", ".opus", ".wav", ".aac", ".wma", ".mp4", ".m4b", ".aiff"}


def fmt_lrc(seconds: float) -> str:
    """seconds -> mm:ss.xx (LRC centisecond format)."""
    if seconds < 0:
        seconds = 0.0
    m = int(seconds // 60)
    s = seconds - m * 60
    return f"{m:02d}:{s:05.2f}"


def render_simple(words: list) -> str:
    lines = []
    for _seg, w in words:
        text = w.word.strip()
        if text:
            lines.append(f"[{fmt_lrc(w.start)}]{text}")
    return "\n".join(lines)


def render_enhanced(groups: list) -> str:
    """groups: list of (line_start, [word,...]) — one timestamped line per whisper segment."""
    out = []
    for line_start, words in groups:
        if not words:
            continue
        parts = []
        for w in words:
            text = w.word.strip()
            if text:
                parts.append(f"<{fmt_lrc(w.start)}>{text}")
        if parts:
            out.append(f"[{fmt_lrc(line_start)}]{' '.join(parts)}")
    return "\n".join(out)


def build_groups(segments) -> list:
    """Collect (segment, word) pairs and pre-group them by segment for enhanced mode."""
    pairs, groups = [], []
    for seg in segments:
        seg_words = [w for w in (seg.words or []) if w.word and w.word.strip()]
        if seg_words:
            groups.append((seg.start, seg_words))
        pairs.extend((seg, w) for w in seg_words)
    return pairs, groups


def _run_ids(pairs: list, max_gap: float) -> list:
    """Parallel list of lyric-run ids: same id when gap to the previous word
    is <= max_gap. pairs must be sorted by word start."""
    ids = [0] if pairs else []
    for i in range(1, len(pairs)):
        gap = pairs[i][1].start - pairs[i - 1][1].end
        ids.append(ids[-1] if gap <= max_gap else ids[-1] + 1)
    return ids


def drop_isolated(pairs: list, max_gap: float, min_run: int) -> list:
    """Remove words that are not part of a contiguous lyric run.

    Whisper hallucinates sparse filler words ('Thank you.', 'yeah') during
    instrumental passages. Real lyrics cluster into dense runs (gap between
    consecutive words << max_gap). Words belonging to runs shorter than
    min_run are dropped. pairs must be sorted by word start.
    """
    if not pairs:
        return pairs
    from collections import Counter
    sizes = Counter(_run_ids(pairs, max_gap))
    return [p for p, rid in zip(pairs, _run_ids(pairs, max_gap)) if sizes[rid] >= min_run]


def smart_filter(pairs: list, max_gap: float, min_run: int,
                 iso_prob: float, run_prob: float) -> list:
    """Confidence filter that does NOT punch holes in real lyric lines.

    The old behavior dropped every word below iso_prob globally — including
    mumbled-but-real sung words in the middle of dense lines. Instead:

      1. Compute lyric "runs" over ALL words (dense clusters whose word gaps
         stay under max_gap). Runs shorter than min_run are whisper's isolated
         hallucinated fills ('Thank you.', 'yeah').
      2. Words inside dense runs survive unless probability < run_prob
         (default 0.05 — near-junk only).
      3. Sparse words survive only if confident (probability >= iso_prob).

    Kills hallucinated fills while keeping low-confidence REAL words inside
    lyric phrases. pairs must be sorted by word start.
    """
    if not pairs:
        return pairs
    from collections import Counter
    ids = _run_ids(pairs, max_gap)
    sizes = Counter(ids)
    kept = []
    for pair, rid in zip(pairs, ids):
        w = pair[1]
        threshold = run_prob if sizes[rid] >= min_run else iso_prob
        if w.probability >= threshold:
            kept.append(pair)
    return kept


GAP_SHIFT_STAMP = "# lrcgen-gap-align:v1"
GAP_THRESHOLD_S = 2.0          # gap (start-to-start) that resets the lyric phrase
GAP_SHIFTS = (0.45, 0.15, 0.05)  # measured whisper earliness for 1st/2nd/3rd word after a gap


def apply_gap_shift(pairs: list, threshold: float = GAP_THRESHOLD_S,
                    shifts: tuple = GAP_SHIFTS) -> bool:
    """Delay words that follow a lyric pause.

    Whisper's word timestamps run ~0.45s early for the first word after a
    gap (measured against professionally-timed lyrics on 18 tracks: median
    -0.44s for word 1, -0.14s for word 2, -0.06s for word 3; ~-0.1s for the
    rest). This shifts the first three words after each gap by those amounts
    so karaoke highlighting no longer jumps ahead of the sung word.

    pairs: list of (segment, word) sorted by word start. Mutates word times
    in place (monotonicity preserved). Returns True if anything shifted.
    """
    if not pairs:
        return False
    # compute per-word shift from ORIGINAL times
    n = len(pairs)
    phrase_pos = [0] * n
    prev_start = None
    for i, (_seg, w) in enumerate(pairs):
        if prev_start is None or w.start - prev_start > threshold:
            phrase_pos[i] = 1
        else:
            phrase_pos[i] = min(phrase_pos[i - 1] + 1, len(shifts) + 1)
        prev_start = w.start
    shift_sec = [shifts[p - 1] if 0 < p <= len(shifts) else 0.0 for p in phrase_pos]
    if not any(shift_sec):
        return False
    # apply, preserving monotonic word-START order (karaoke lights words by start)
    last_new_start = None
    for i, (seg, w) in enumerate(pairs):
        d = shift_sec[i]
        if d > 0:
            w.start += d
            w.end += d
        if last_new_start is not None and w.start < last_new_start:
            w.start = last_new_start
            w.end = max(w.end, w.start)
        last_new_start = w.start
    return True


def render_official(groups) -> str:
    """groups: [(line_start, [(sec, word_str), ...])] -> enhanced LRC text."""
    out = []
    for start, wt in groups:
        parts = [f"<{fmt_lrc(t)}>{w}" for t, w in wt]
        out.append(f"[{fmt_lrc(start)}]{' '.join(parts)}")
    return "\n".join(out)


def try_official_lyrics(f: Path, duration: float, hyp_words):
    """Fetch official lyrics and align them to whisper's words.

    Returns enhanced-LRC body text, or None when no lyrics are found or the
    alignment is too weak to trust (falls back to pure transcription).
    """
    import lyricsync
    tags = lyricsync.read_tags(f)
    if not tags.get("title") or not tags.get("artist"):
        return None
    rec = lyricsync.fetch_lrclib(tags["artist"], tags["title"],
                                 tags.get("album"), duration)
    if not rec or rec.get("instrumental"):
        return None
    synced_times = None
    ref_lines = None
    if rec.get("syncedLyrics"):
        synced_times = lyricsync.parse_synced(rec["syncedLyrics"])
        if synced_times:
            ref_lines = [t for _, t in synced_times]
    if ref_lines is None and rec.get("plainLyrics"):
        ref_lines = lyricsync.split_lines(rec["plainLyrics"])
    if not ref_lines:
        return None
    res = lyricsync.align(ref_lines, hyp_words, synced_times)
    if res is None:
        print("  official lyrics found but alignment too weak; transcribed text used", flush=True)
        return None
    groups, ratio = res
    return render_official(groups), ratio, len(ref_lines)


def metadata_header(path: Path) -> str:
    """Best-effort [ti:]/[ar:]/[al:] header from embedded tags (mutagen)."""
    try:
        from mutagen import File as MFile
        mf = MFile(path)
        if mf is None:
            return ""
        def get(*keys):
            for k in keys:
                v = mf.get(k)
                if v:
                    return str(v[0] if isinstance(v, list) else v)
            return ""
        parts = []
        for tag, keys in (("ti", ("title", "TIT2")), ("ar", ("artist", "TPE1")), ("al", ("album", "TALB"))):
            val = get(*keys)
            if val:
                parts.append(f"[{tag}:{val}]")
        return "\n".join(parts) + "\n" if parts else ""
    except Exception:
        return ""


def separate_vocals(path: Path, workdir: Path, model_name: str) -> Path:
    """Isolate vocals with demucs so whisper hears lyrics, not the full mix.

    Returns the vocal stem path on the SAME timeline as the original audio
    (word timestamps remain valid for the sidecar .lrc).
    """
    import subprocess
    import shutil
    out_root = workdir / "lrcgen-demucs"
    out_root.mkdir(parents=True, exist_ok=True)
    import shutil as _sh
    demucs_bin = _sh.which("demucs") or str(Path(sys.executable).parent / "demucs")
    cmd = [demucs_bin, "-n", model_name, "--two-stems=vocals", "-o", str(out_root), str(path)]
    # torch must not see this venv's pip-wheel CUDA/cuDNN libs (exported for
    # ctranslate2): mixed cuDNN sublibrary versions crash conv1d. Point it at
    # the consistent system CUDA libraries instead.
    env = {k: v for k, v in os.environ.items()}
    env["LD_LIBRARY_PATH"] = "/usr/lib"
    print("  demucs: separating vocals ...", flush=True)
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if r.returncode != 0:
        raise RuntimeError(f"demucs failed: {r.stderr.strip()[-400:]}")
    stems = list(out_root.rglob("vocals.wav"))
    if not stems:
        raise RuntimeError("demucs produced no vocals.wav")
    print(f"  demucs: done in {time.time() - t0:.0f}s", flush=True)
    return stems[0]


def collect_audio(paths: list[Path]) -> list[Path]:
    files = []
    for p in paths:
        if p.is_dir():
            files.extend(
                f for f in sorted(p.rglob("*"))
                if f.is_file() and f.suffix.lower() in AUDIO_EXT
                and not f.name.startswith(".") and not any(part.startswith(".") for part in f.parts)
            )
        elif p.is_file() and p.suffix.lower() in AUDIO_EXT:
            files.append(p)
    # dedupe, keep order
    seen, out = set(), []
    for f in files:
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", type=Path, help="audio file(s) and/or directories to process (recursive)")
    ap.add_argument("--model", default="large-v3-turbo",
                    help="faster-whisper model id or size (default: large-v3-turbo; try large-v3 for max accuracy)")
    ap.add_argument("--device", default="cuda", help="cuda | cpu (default: cuda)")
    ap.add_argument("--compute-type", default=None, help="float16 (default on cuda), int8_float16, int8 on cpu, ...")
    ap.add_argument("--format", choices=["simple", "enhanced"], default="simple",
                    help="lrc layout: word-per-line (simple) or karaoke inline timestamps (enhanced)")
    ap.add_argument("--language", default=None, help="force language code (default: auto-detect per file)")
    ap.add_argument("--skip-existing", action="store_true",
                    help="skip files that already have a .lrc (default: REGENERATE all — existing "
                         ".lrc files are not assumed correct)")
    ap.add_argument("--skip-from-log", type=Path, default=None,
                    help="resume file: skip every path listed with an OK or SKIP result in this log")
    ap.add_argument("--vad", action="store_true",
                    help="enable Silero VAD speech filtering (default OFF: VAD suppresses vocals over music; "
                         "use for podcasts/spoken audio)")
    ap.add_argument("--min-word-prob", type=float, default=0.3,
                    help="confidence floor for SPARSE words outside lyric runs "
                         "(default 0.3; 0 disables). Words inside dense lyric runs are "
                         "governed by --run-min-word-prob instead")
    ap.add_argument("--run-min-word-prob", type=float, default=0.05,
                    help="confidence floor for words INSIDE dense lyric runs "
                         "(default 0.05). Low because mumbled-but-real sung words often "
                         "score low; raising it removes more uncertain words")
    ap.add_argument("--no-speech-threshold", type=float, default=0.9,
                    help="whisper skips a segment when its no-speech probability exceeds "
                         "this AND the segment logprob is poor. Whisper's default 0.6 drops "
                         "real sung lines over loud instrumentation; music wants a high "
                         "value (default 0.9). Pass 0 to disable segment skipping entirely")
    ap.add_argument("--log-prob-threshold", type=float, default=-1.0,
                    help="whisper's average-logprob fallback threshold (default -1.0)")
    ap.add_argument("--initial-prompt", default=None,
                    help="optional text prompt that biases transcription style, e.g. song "
                         "title/artist or a few real lyric lines (helps proper nouns and slang)")
    ap.add_argument("--fetch-lyrics", action="store_true",
                    help="look up official lyrics on LRCLIB (free, no key) and align them "
                         "to whisper's timing instead of trusting the transcription — near-"
                         "perfect word accuracy when the track is in the database")
    ap.add_argument("--demucs", action="store_true",
                    help="isolate vocals with demucs before transcribing (much better word "
                         "accuracy on busy mixes; adds GPU/CPU cost per track)")
    ap.add_argument("--demucs-model", default="htdemucs",
                    help="demucs model (default htdemucs)")
    ap.add_argument("--min-words", type=int, default=15,
                    help="tracks with fewer surviving words than this are treated as instrumental and skipped")
    ap.add_argument("--condition-on-previous-text", action="store_true",
                    help="let whisper condition on previous segment text (default OFF; reduces repetition loops on music)")
    ap.add_argument("--max-gap", type=float, default=4.0,
                    help="gap (s) that separates lyric runs in the island-removal filter")
    ap.add_argument("--min-run", type=int, default=2,
                    help="drop runs shorter than this many words (isolated hallucinated fills)")
    ap.add_argument("--keep-isolated", action="store_true",
                    help="disable island-removal (keeps every word that passes the probability filter)")
    ap.add_argument("--no-gap-shift", action="store_true",
                    help="disable the lyric-gap timing correction (whisper is ~0.45s early "
                         "for the first word after a pause)")
    ap.add_argument("--beam-size", type=int, default=5)
    ap.add_argument("--dry-run", action="store_true", help="only list files that would be processed")
    ap.add_argument("--log-file", type=Path, default=None, help="append progress lines to this file")
    args = ap.parse_args()

    if args.compute_type is None:
        # int8_float16: weights in int8 (≈0.8 GB for turbo) — far less likely to OOM
        # when the desktop/games share the GPU; quality loss vs float16 is negligible.
        args.compute_type = "int8_float16" if args.device == "cuda" else "int8"

    files = collect_audio(args.paths)
    # resume tracking is path + mtime: a replaced/re-ripped file (same path,
    # different audio) no longer matches its old log entry and gets redone
    done_from_log = {}
    if args.skip_from_log is not None:
        try:
            for line in args.skip_from_log.read_text(encoding="utf-8", errors="replace").splitlines():
                parts = line.split("\t")
                if len(parts) >= 2 and parts[0] in ("OK", "SKIP"):
                    mt = next((float(p[6:]) for p in parts if p.startswith("mtime=")), None)
                    done_from_log.setdefault(parts[1], []).append(mt)
            print(f"resume log: {sum(len(v) for v in done_from_log.values())} "
                  f"already-done tracks will be skipped (path+mtime matched)")
        except FileNotFoundError:
            print(f"warning: resume log {args.skip_from_log} not found; processing everything")

    def is_done(f: Path) -> bool:
        entries = done_from_log.get(str(f))
        if not entries:
            return False
        cur = f.stat().st_mtime
        return any(m is None or abs(cur - m) < 0.001 for m in entries)

    todo = []
    for f in files:
        lrc = f.with_suffix(".lrc")
        if lrc.exists() and args.skip_existing:
            continue
        if is_done(f):
            continue
        todo.append(f)

    skipped_reason = "already in resume log" if args.skip_from_log else "already have .lrc and --skip-existing"
    print(f"found {len(files)} audio file(s); {len(todo)} to (re)generate "
          f"({len(files) - len(todo)} skipped: {skipped_reason})")
    if args.dry_run:
        for f in todo:
            print("  would process:", f)
        return 0
    if not todo:
        return 0

    import gc
    from faster_whisper import WhisperModel

    def load_model():
        t_load = time.time()
        print(f"loading model '{args.model}' on {args.device} ({args.compute_type}) ...", flush=True)
        m = WhisperModel(args.model, device=args.device, compute_type=args.compute_type)
        print(f"model ready in {time.time() - t_load:.1f}s", flush=True)
        return m

    model = load_model()

    def transcribe_with_retry(audio_path):
        """Transcribe with self-healing: GPU hiccups (OOM, device loss while the
        desktop/games share VRAM) are recovered by reloading the model, up to 3 tries."""
        nonlocal model
        for attempt in range(1, 4):
            try:
                # transcribe() returns a lazy generator: force iteration inside the
                # retry scope so word-timestamp errors surface here, not in main().
                segments, info = model.transcribe(
                    str(audio_path),
                    language=args.language,
                    beam_size=args.beam_size,
                    word_timestamps=True,
                    vad_filter=args.vad,
                    condition_on_previous_text=args.condition_on_previous_text,
                    initial_prompt=args.initial_prompt,
                    no_speech_threshold=(None if args.no_speech_threshold == 0
                                         else args.no_speech_threshold),
                    log_prob_threshold=args.log_prob_threshold,
                )
                segs = list(segments)
                return segs, info
            except Exception as e:
                msg = str(e).lower()
                is_gpu = any(k in msg for k in ("cuda", "out of memory", "invalid device", "cublas", "cudnn", "driver"))
                if is_gpu and attempt < 3:
                    print(f"  RECOVER: GPU error ({e}); reloading model, retry {attempt}/3", flush=True)
                    del model
                    gc.collect()
                    time.sleep(3)
                    model = load_model()
                    continue
                if attempt == 1 and "boolean index" in msg:
                    # known faster-whisper word-timestamp bug (empty segment); retry once via VAD path
                    print(f"  RECOVER: alignment bug ({e}); retrying with VAD path", flush=True)
                    segments2, info2 = model.transcribe(
                        str(path), language=args.language, beam_size=args.beam_size,
                        word_timestamps=True, vad_filter=True,
                        condition_on_previous_text=args.condition_on_previous_text,
                        initial_prompt=args.initial_prompt,
                        no_speech_threshold=(None if args.no_speech_threshold == 0
                                             else args.no_speech_threshold),
                        log_prob_threshold=args.log_prob_threshold,
                    )
                    return list(segments2), info2
                raise
        raise RuntimeError("unreachable")

    logf = open(args.log_file, "a", buffering=1) if args.log_file else None
    total_wall = time.time()
    done = skipped = failed = 0
    for f in todo:
        lrc = f.with_suffix(".lrc")
        t0 = time.time()
        vocal_stem = None
        try:
            if args.demucs:
                vocal_stem = separate_vocals(f, f.parent, args.demucs_model)
            segments, info = transcribe_with_retry(vocal_stem or f)
            pairs, _ = build_groups(segments)
            # one confidence/island pass over ALL words: dense lyric runs keep
            # even low-confidence (mumbled but real) words; sparse words must
            # be confident and belong to a run of >= min_run words
            pairs = smart_filter(pairs, args.max_gap, args.min_run,
                                 iso_prob=(args.min_word_prob if args.min_word_prob > 0 else 0.0),
                                 run_prob=(args.run_min_word_prob if args.run_min_word_prob > 0 else 0.0))
            official = None
            if args.fetch_lyrics and args.format == "enhanced":
                official = try_official_lyrics(f, info.duration, [w for _, w in pairs])
            # rebuild segment groups from the surviving words so enhanced
            # mode benefits from island removal too
            groups = []
            for seg, w in pairs:
                if groups and groups[-1][0] is seg:
                    groups[-1][1].append(w)
                else:
                    groups.append([seg, [w]])
            groups = [(seg.start, words) for seg, words in groups]
            if not args.no_gap_shift:
                shifted = apply_gap_shift(pairs)
                # re-anchor groups on the (possibly shifted) first word and
                # refresh line starts so the pane doesn't switch lines early
                groups = []
                for seg, w in pairs:
                    if groups and groups[-1][0] is seg:
                        groups[-1][1].append(w)
                    else:
                        groups.append([seg, [w]])
                groups = [(words[0].start if words else seg.start, words) for seg, words in groups]
            if not pairs or len(pairs) < args.min_words:
                print(f"  SKIP (only {len(pairs)} words survive probability filter, likely instrumental): {f}")
                skipped += 1
                if logf:
                    logf.write(f"SKIP\t{f}\t{len(pairs)} words\tmtime={f.stat().st_mtime:.6f}\n")
                continue
            if official is not None:
                body, ratio, n_lines = official
                body = body + f"\n# aligned {ratio*100:.0f}% of {n_lines} official lines"
                header = metadata_header(f) + "# lrcgen-official-lyrics:v1\n"
            else:
                body = render_simple(pairs) if args.format == "simple" else render_enhanced(groups)
                header = metadata_header(f)
                if not args.no_gap_shift:
                    header += GAP_SHIFT_STAMP + "\n"
            content = header + body + "\n"
            # atomic write: a kill mid-batch (nightly time window) must not
            # leave a truncated .lrc that players would show as broken lyrics
            tmp = lrc.with_suffix(".lrc.tmp")
            tmp.write_text(content, encoding="utf-8")
            os.replace(tmp, lrc)
            dt = time.time() - t0
            speed = info.duration / dt if dt > 0 else 0
            done += 1
            src_tag = "official" if official is not None else "transcribed"
            line = (f"  OK  {f.name}  [{src_tag}] lang={info.language} "
                    f"conf={info.language_probability:.2f} "
                    f"dur={info.duration:.0f}s words={len(pairs)} {dt:.1f}s ({speed:.1f}x realtime)")
            print(line, flush=True)
            if logf:
                logf.write(f"OK\t{f}\t{info.language}\t{info.duration:.1f}\t{len(pairs)}\t{dt:.1f}"
                           f"\tmtime={f.stat().st_mtime:.6f}\n")
        except Exception as e:
            failed += 1
            msg = f"  FAIL {f}: {e}"
            print(msg, flush=True)
            if logf:
                logf.write(f"FAIL\t{f}\t{e}\n")
        finally:
            if vocal_stem is not None:
                import shutil
                shutil.rmtree(vocal_stem.parent.parent, ignore_errors=True)

    total = time.time() - total_wall
    print(f"\nfinished: {done} ok, {skipped} skipped (instrumental), {failed} failed, {total:.0f}s total")
    if logf:
        logf.write(f"SUMMARY\tok={done}\tskip={skipped}\tfail={failed}\tsecs={total:.0f}\n")
        logf.close()
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
