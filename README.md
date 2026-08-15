# s2udio-lrc — word-timed .lrc sidecars for a local music library

Generate **word-per-word lyric `.lrc` files** for your own music, fully
locally — a local Whisper model (via `faster-whisper`) transcribes and
word-times every track on your GPU. No cloud, no uploads.

Built for and verified against [s2udio](https://github.com/NJMRgit/s2udio)'s
lyrics pane: the output is **enhanced LRC** (`[mm:ss.xx]<mm:ss.xx>word …`),
which s2udio (and most karaoke players) highlight word-by-word in time.

## Features

- **Word-level timing** from Whisper's own word timestamps (large-v3-turbo
  by default; `--model large-v3` for maximum accuracy).
- **Enhanced karaoke format**: one line per lyric phrase, each word with its
  own `<mm:ss.xx>` marker. Plain word-per-line output also available
  (`--format simple`).
- **Instrumental detection** — tracks with no real vocals are skipped
  (verified against a 1,400+ track cross-check: zero false skips on tracks
  that have lyrics anywhere).
- **Hallucination cleanup**: probability filter + "island removal" drop
  whisper's fake fill words (`Thank you.`, `yeah`) on instrumental breaks.
- **Crash-safe batch runs**: per-track tab-separated log, `--resume`
  re-processes only what failed or was interrupted.
- **Self-healing**: GPU OOM / device-loss (games or desktop apps sharing the
  card) reloads the model and retries; a known faster-whisper
  "boolean index" word-timestamp bug retries via the VAD path.

## Requirements

- Linux; any NVIDIA GPU with CUDA 12 works — the model needs only ~1–2 GB of
  VRAM in the default `int8_float16` mode (a 4 GB card is plenty). No GPU?
  CPU mode works too (`--device cpu`) but is roughly 10–20× slower.
- Python **3.12** (ctranslate2 — faster-whisper's engine — has no wheels for
  Python 3.14+ yet).
- `ffmpeg`/`ffprobe` on PATH (used by the audio analysis).

## Install

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

The first run downloads the Whisper model (~1.6 GB for
`large-v3-turbo`) into `~/.cache/huggingface`.

If the system CUDA is not 12.x, install the CUDA 12 runtime libraries as
pip wheels (the `lrcgen` launcher exposes them to the dynamic linker
automatically):

```bash
.venv/bin/pip install nvidia-cublas-cu12 nvidia-cudnn-cu12 nvidia-cuda-runtime-cu12
```

## Quick start

```bash
# One file
./lrcgen "/path/to/01 - Track.flac" --format enhanced

# A whole directory (recursive), enhanced karaoke format
./lrcgen "/mnt/Music/Artist/Album" --format enhanced

# Preview without transcribing
./lrcgen /path/to/music --dry-run
```

Each audio file gets a `.lrc` sidecar next to it
(`01 - Track.flac` → `01 - Track.lrc`). **Existing `.lrc` files are not
trusted**: by default every track is re-transcribed and its `.lrc`
rewritten. Use `--skip-existing` to leave existing files alone.

## Batch runs (whole library)

```bash
./run_library.sh --enhanced            # everything, enhanced format
./run_library.sh --enhanced --resume   # continue after interruption
./run_library.sh --enhanced --dry-run  # list what would be processed
```

- The library root defaults to `/mnt/20TBHDD/Media/Music`; override with
  `LRCGEN_LIB=/path/to/music ./run_library.sh …`.
- Progress is appended to `library.log` (simple) / `library-enhanced.log`
  (enhanced), one tab-separated line per track: `OK|SKIP|FAIL <path> …`.
- Resume skips only tracks logged `OK`/`SKIP`; anything else (including
  crash-truncated `.lrc` files) is redone.
- A typical 12k-track library runs in ~7–10 h on the development machine
  (RTX 4080: ~2–5 s/track, 50–120× realtime); on weaker GPUs expect
  proportionally longer, and CPU mode is ~10–20× slower again.

## Fixing timings on already-generated files

Whisper's gap bias is corrected at generation time, but if you changed the
tuning (or want to re-run the correction on existing files without
re-transcribing), `gap_align.py` applies the same rule to enhanced `.lrc`
files in place (idempotent, stamped `# lrcgen-gap-align:v1`):

```bash
./gap_align.py /path/to/music            # fix everything under the dir
./gap_align.py one/file.lrc              # or single files
```

## Tuning

| Flag | Default | Purpose |
|---|---|---|
| `--model` | `large-v3-turbo` | Whisper size (`large-v3` = max accuracy, ~4× slower) |
| `--format` | `simple` | `enhanced` = karaoke lines with inline word markers (recommended for s2udio) |
| `--min-word-prob` | `0.3` | drop words below this token probability (hallucination filter) |
| `--min-words` | `15` | tracks with fewer surviving words are skipped as instrumental |
| `--keep-isolated` | off | disable island removal (keeps every word that passes the probability filter) |
| `--max-gap` / `--min-run` | `4.0` / `2` | island-removal knobs: words in runs shorter than `min-run`, gapped by >`max-gap`s, are dropped |
| `--no-gap-shift` | off | disable the pause-timing correction |
| `--vad` | off | Silero VAD filtering — for podcasts/spoken audio; **off by default** because it suppresses vocals over music |
| `--skip-existing` | off | keep existing `.lrc` files instead of regenerating |
| `--skip-from-log FILE` | — | resume: skip only tracks logged `OK`/`SKIP` in FILE |
| `--language` | auto | force a language code |
| `--compute-type` | `int8_float16` (cuda) | half the VRAM of `float16`, negligible quality loss |

## s2udio integration

s2udio reads `.lrc` sidecars from the MPD library as its **priority-1,
read-only** lyric source (`Ctx::find_current_lyrics_path`), before the
`lyrics_dir` mirror and the lyrics index, so generated sidecars show up
automatically on the next song change — no config, no restart. If you use
`rmpc-fetch-lyrics` as the `on_song_change` hook, it already skips fetching
when a sidecar exists (`HAS_LRC=true` guard).

## Troubleshooting

- **`libcublas.so.12 not found`** — install the `nvidia-*-cu12` pip wheels
  (above) and launch via `./lrcgen` (not `python lrcgen.py`) so the CUDA-12
  libs are on `LD_LIBRARY_PATH`.
- **Slow batch while the desktop is busy** — whisper shares the GPU with
  games/video; the batch self-heals on OOM but runs slower. Let it run when
  the GPU is otherwise idle, or resume later.
- **`boolean index did not match indexed array`** — known faster-whisper
  word-timestamp edge case; lrcgen retries such tracks via the VAD path
  automatically.
- **Wrong language auto-detect** — pass `--language <code>` (e.g. `en`,
  `ru`).
