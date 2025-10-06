#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build/merge a SymSpell-compatible frequency list from your corpus.
- Accepts a directory or file (--in).
- Reads .txt, .md, and .json (extracts string values).
- Merges with an existing frequency file (--merge) if provided.
- Writes token<TAB>count sorted by count desc.

Usage:
  python processors/tools/build_frequency_list.py \
      --in /path/to/corpus_or_file \
      --merge processors/data/frequency_en_82k.txt \
      --out processors/data/frequency_en_82k.txt \
      --top 82000
"""
import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9\-']{0,}", flags=re.UNICODE)

def iter_texts(path: Path):
    if path.is_dir():
        for p in path.rglob("*"):
            if not p.is_file():
                continue
            if p.suffix.lower() in {".txt", ".md"}:
                try:
                    yield p.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    continue
            elif p.suffix.lower() == ".json":
                try:
                    data = json.loads(p.read_text(encoding="utf-8", errors="ignore"))
                    yield from extract_strings(data)
                except Exception:
                    continue
    else:
        if path.suffix.lower() in {".txt", ".md"}:
            yield path.read_text(encoding="utf-8", errors="ignore")
        elif path.suffix.lower() == ".json":
            try:
                data = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
                yield from extract_strings(data)
            except Exception:
                pass

def extract_strings(obj):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from extract_strings(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from extract_strings(v)

def tokenize(text: str):
    # Normalize some punctuation & case
    text = text.replace("—", "-").replace("–", "-").replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    for m in TOKEN_RE.finditer(text):
        tok = m.group(0)
        if tok:
            yield tok.lower()

def load_freq(path: Path):
    freq = Counter()
    if not path.exists():
        return freq
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "\t" not in line:
                # allow "token count" too (space separated)
                parts = line.split()
                if len(parts) >= 2 and parts[-1].isdigit():
                    token = " ".join(parts[:-1])
                    count = int(parts[-1])
                    freq[token] += count
                continue
            token, count = line.split("\t", 1)
            token = token.strip()
            count = count.strip()
            try:
                freq[token] += int(count)
            except Exception:
                continue
    return freq

def save_freq(freq: Counter, out_path: Path, top: int = 0):
    items = freq.most_common(top or None)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for token, count in items:
            if not token:
                continue
            f.write(f"{token}\t{count}\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="Input file or directory (corpus)")
    ap.add_argument("--merge", dest="merge", default="", help="Existing frequency file to merge (optional)")
    ap.add_argument("--out", dest="out", required=True, help="Output frequency file")
    ap.add_argument("--top", dest="top", type=int, default=82000, help="Keep only top-N (0 = all)")
    args = ap.parse_args()

    inp = Path(args.inp)
    outp = Path(args.out)
    mergep = Path(args.merge) if args.merge else None

    # 1) start from existing list if provided
    freq = Counter()
    if mergep:
        freq |= load_freq(mergep)

    # 2) ingest corpus
    if not inp.exists():
        print(f"[!] Input path not found: {inp}", file=sys.stderr)
        sys.exit(1)

    corpus_counts = Counter()
    docs = 0
    for txt in iter_texts(inp):
        docs += 1
        corpus_counts.update(tokenize(txt))

    if docs == 0:
        print("[i] No documents found to ingest; will only rewrite/keep merged list.", file=sys.stderr)

    # 3) merge & write
    freq |= corpus_counts
    save_freq(freq, outp, top=args.top)
    print(f"[ok] Wrote {outp} ({len(freq)} unique tokens, top={args.top or 'all'})")

if __name__ == "__main__":
    main()
