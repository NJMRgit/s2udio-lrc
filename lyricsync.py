#!/usr/bin/env python3
"""Official-lyrics fetching + alignment for lrcgen.

Instead of trusting whisper's transcription for WORDS, fetch the official
lyrics (LRCLIB, free API) and use whisper only as a TIMING anchor: fuzzy-align
the known lyric lines onto whisper's word stream and interpolate word
timestamps. Text accuracy = the official lyrics themselves; whisper errors no
longer corrupt the words.
"""
from __future__ import annotations

import difflib
import re
from pathlib import Path

import requests

UA = "s2udio-lrc/1.0 (local karaoke lyric generator)"
API = "https://lrclib.net/api"


def _norm(word: str) -> str:
    return re.sub(r"[^\w']", "", word.lower())


def read_tags(path: Path) -> dict:
    """Best-effort {title, artist, album} from embedded tags."""
    try:
        from mutagen import File as MFile
        mf = MFile(str(path))
        if mf is None:
            return {}
        def get(*keys):
            for k in keys:
                v = mf.get(k)
                if v:
                    return str(v[0] if isinstance(v, list) else v).strip()
            return ""
        tags = {"title": get("title", "TIT2"),
                "artist": get("artist", "TPE1"),
                "album": get("album", "TALB")}
        # untagged or useless title ("Track01", "Track 3"): fall back to the
        # filename, e.g. "01 - Smells Like Teen Spirit" -> "Smells Like Teen Spirit"
        if re.fullmatch(r"(?i)(track\s*\d*|untitled|\d{1,3})\s*", tags["title"] or ""):
            tags["title"] = ""
            parts = [p.strip() for p in re.split(r"\s+-\s+", path.stem) if p.strip()]
            parts = [p for p in parts if not re.fullmatch(r"\d{1,3}[.)]?", p)]
            if parts:
                tags["title"] = parts[-1]
                if not tags["artist"] and len(parts) == 3:
                    tags["artist"] = parts[0]
        return tags
    except Exception:
    # strip "(feat. ...)", "- Remastered 2011" style noise for lookups
        return {}


def clean_lookup(s: str) -> str:
    s = re.sub(r"\(feat\.[^)]*\)", "", s, flags=re.I)
    s = re.sub(r"\(ft\.[^)]*\)", "", s, flags=re.I)
    s = re.sub(r"\[[^\]]*\]", "", s)
    s = re.sub(r"\s*-\s*(remaster|live|demo|version|mix|edit|remix|mono|stereo).*$", "",
               s, flags=re.I)
    return s.strip()


def fetch_lrclib(artist: str, title: str, album: str | None,
                 duration: float | None) -> dict | None:
    """Return the best LRCLIB record ({syncedLyrics, plainLyrics, ...}) or None."""
    try:
        params = {"artist_name": clean_lookup(artist),
                  "track_name": clean_lookup(title)}
        if album:
            params["album_name"] = clean_lookup(album)
        if duration:
            params["duration"] = str(int(round(duration)))
        r = requests.get(f"{API}/get", params=params, timeout=15, headers={"User-Agent": UA})
        if r.status_code == 200:
            rec = r.json()
            if rec.get("plainLyrics") or rec.get("syncedLyrics"):
                return rec
        r = requests.get(f"{API}/search",
                         params={"track_name": params["track_name"],
                                 "artist_name": params["artist"]},
                         timeout=15, headers={"User-Agent": UA})
        hits = r.json() if r.status_code == 200 else []
        def score(h):
            s = 0
            if duration and h.get("duration"):
                s -= abs(h["duration"] - duration)
            s += 5 * bool(h.get("syncedLyrics"))
            return s
        for h in sorted(hits, key=score, reverse=True):
            if h.get("plainLyrics") or h.get("syncedLyrics"):
                if duration and h.get("duration") and abs(h["duration"] - duration) > 20:
                    continue
                return h
    except Exception:
        return None
    return None


def parse_synced(synced: str):
    """LRCLIB synced text -> [(sec, line)], dropping empty/section-marker lines."""
    out = []
    pat = re.compile(r"^\[(\d+):(\d+(?:\.\d+)?)\](.*)$")
    for ln in synced.splitlines():
        m = pat.match(ln.strip())
        if not m:
            continue
        t = int(m.group(1)) * 60 + float(m.group(2))
        text = m.group(3).strip()
        if text and not re.fullmatch(r"[\[\(].*[\]\)]", text):
            out.append((t, text))
    return out or None


def split_lines(text: str):
    """Plain lyrics -> [line], dropping empties and [section] markers."""
    out = []
    for ln in text.splitlines():
        ln = ln.strip()
        if ln and not re.fullmatch(r"[\[\(].*[\]\)]", ln):
            out.append(ln)
    return out


def align(ref_lines: list[str], hyp_words: list, synced_times=None):
    """Align official lyric lines to whisper words.

    ref_lines:     official lyric line strings
    hyp_words:     faster-whisper Word objects (.word/.start/.end)
    synced_times:  optional [(sec,line)] human line timings

    Returns (groups, ratio): groups = [(line_start_sec, [(sec, word_str), ...])]
    or None if too little aligned. ratio = fraction of ref words matched.
    """
    ref_words = []       # (line_idx, word_str)
    for li, ln in enumerate(ref_lines):
        for w in ln.split():
            ref_words.append((li, w))
    ref_norms = [_norm(w) for _, w in ref_words]
    hyp_norms = [_norm(w.word) for w in hyp_words]
    if not ref_norms or not hyp_norms:
        return None

    sm = difflib.SequenceMatcher(None, hyp_norms, ref_norms, autojunk=False)
    ref2hyp = {}
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                ref2hyp[j1 + k] = i1 + k
    ratio = len(ref2hyp) / len(ref_norms)

    # per-line matched anchors: [(pos_in_line, hyp_time)]
    line_word_pos = {}
    for gi, (li, _) in enumerate(ref_words):
        line_word_pos.setdefault(li, []).append(gi)

    synced_map = {}
    if synced_times:
        # fuzzy-match synced line text to our line strings for their times
        norm_lines = [_norm(x) for x in ref_lines]
        for t, sl in synced_times:
            sn = _norm(sl.replace(",", " "))
            # find best containment
            best_i, best_r = None, 0.0
            for i, ln in enumerate(ref_lines):
                a, b = sn, _norm(ln)
                if not a or not b:
                    continue
                r = difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()
                if a in b or b in a:
                    r += 0.5
                if r > best_r:
                    best_i, best_r = i, r
            if best_i is not None and best_r > 0.8 and best_i not in synced_map:
                synced_map[best_i] = t

    # gather anchors per line
    line_anchors = {}   # li -> [(pos, sec)]
    for li, gis in line_word_pos.items():
        anch = []
        for pos, gi in enumerate(gis):
            hi = ref2hyp.get(gi)
            if hi is not None:
                anch.append((pos, float(hyp_words[hi].start)))
        line_anchors[li] = anch

    total_matched = sum(1 for a in line_anchors.values() for _ in a)
    if total_matched < max(5, 0.25 * len(ref_words)):
        return None

    n_lines = len(ref_lines)
    groups = []
    for li, ln in enumerate(ref_lines):
        words = ln.split()
        anch = sorted(line_anchors[li])
        t_sync = synced_map.get(li)
        if not anch:
            groups.append([None, [(None, w) for w in words], li])   # resolve later
            continue
        times = []
        apos = [p for p, _ in anch]
        for pos in range(len(words)):
            if pos in apos:
                times.append(dict(anch)[pos])
            elif pos < apos[0]:
                back = dict(anch)[apos[0]]
                # walk backwards: assume ~0.35s per preceding word, clamped
                times.append(max(0.0, back - 0.35 * (apos[0] - pos)))
            elif pos > apos[-1]:
                fwd = dict(anch)[apos[-1]]
                times.append(fwd + 0.35 * (pos - apos[-1]))
            else:
                lo_i = max(p for p in apos if p < pos)
                hi_i = min(p for p in apos if p > pos)
                lo_t, hi_t = dict(anch)[lo_i], dict(anch)[hi_i]
                frac = (pos - lo_i) / (hi_i - lo_i)
                times.append(lo_t + frac * (hi_t - lo_t))
        start = min(times)
        if t_sync is not None and abs(t_sync - times[0]) < 6:
            # trust the human line timing when it roughly agrees
            drift = t_sync - times[0]
            times = [max(0.0, t + drift * 0.5) for t in times]
        groups.append([start, list(zip(times, words)), li])

    # resolve unanchored lines: spread between neighbouring resolved lines
    resolved = [g[2] for g in groups if g[0] is not None]
    for idx, g in enumerate(groups):
        if g[0] is not None:
            continue
        li = g[2]
        prev = next((x for x in reversed(resolved) if x < li), None)
        nxt = next((x for x in resolved if x > li), None)
        if synced_map.get(li) is not None:
            g[0] = synced_map[li]
        elif prev is not None and nxt is not None:
            pt = groups[[x[2] for x in groups].index(prev)][1][-1][0]
            nt = groups[[x[2] for x in groups].index(nxt)][1][0][0]
            before = sum(len(groups[k][1]) for k in range(idx) if groups[k][2] is not None and groups[k][2] < li) or 1
            span_words = sum(len(x[1]) for x in groups if prev < x[2] < nxt)
            g[0] = pt + (nt - pt) * (len(g[1]) / max(1, span_words))
        elif prev is None and nxt is not None:
            g[0] = max(0.0, groups[[x[2] for x in groups].index(nxt)][1][0][0] - 0.4 * len(g[1]))
        else:
            g[0] = (groups[[x[2] for x in groups].index(prev)][1][-1][0]
                    if prev is not None else 0.0)
        step = 0.35
        g[1] = [(g[0] + i * step, w) for i, (_, w) in enumerate(g[1])]

    # enforce monotonicity across the whole song
    last = 0.0
    for g in groups:
        fixed = []
        for t, w in g[1]:
            t = max(t, last + 0.01)
            fixed.append((t, w))
            last = t
        g[0] = fixed[0][0]
        g[1] = fixed
    out = [(g[0], g[1]) for g in groups]
    return out, ratio
