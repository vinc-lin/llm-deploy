#!/usr/bin/env python
"""Fetch real outdoor road/weather photographs for the device test kit.

The kit exists so the device team can tell "the pipeline works" from "the
pipeline is fluent and wrong" -- which on this stack is the actual failure
mode. That only works with REAL photographs of the scenes the deployment
cares about: the view out of a vehicle -- road, surroundings, weather. A
synthetic shape on a white background cannot exercise that, and it is also
outside the ViT's COCO-photograph calibration range.

Two sources, both reachable directly (no proxy -- verified 2026-08-15):

  Wikimedia Commons  searched by scene, so weather variety is found rather
                     than guessed from remembered filenames. Freely licensed,
                     and this script records the licence + author for every
                     file it keeps, because these ship inside a public bundle.
  COCO val2017       the same ids the ViT calibration used, as a fallback for
                     generic street scenes. In-calibration by construction.

Every candidate is verified end to end: magic bytes, a real PIL decode, mode,
and a minimum edge length (the pipeline resizes to 512x512, and upscaling a
thumbnail produces a blurred input that tests nothing).

Selection by CONTENT is not done here -- that is the captioning pass in
Phase 4.2, which asks the model what it actually sees. This step only
guarantees a pool of usable, decodable, correctly-licensed photographs.

  $PY_DEPLOY scripts/pipeline/fetch_test_images.py --out work/kit-candidates
"""
import argparse
import json
import socket
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

UA = "llm-deploy-testkit/1.0 (SA8797P device test kit; contact: repo owner)"
COMMONS_API = "https://commons.wikimedia.org/w/api.php"
COCO_URL = "http://images.cocodataset.org/val2017/{:012d}.jpg"

# Scene coverage the kit must have. Phrased as search terms, not filenames.
SCENES = {
    "rain": "rainy road traffic wet street rain",
    "fog": "foggy road fog highway mist",
    "snow": "snowy road winter street snow car",
    "clear": "sunny road highway blue sky landscape",
    "overcast": "overcast cloudy road highway grey sky",
    "street": "city street cars traffic daytime",
}

# Same ids as the ViT calibration set (quantize_vit_aimet.py), so the fallback
# pool is in-distribution for the quantized encoder by construction.
COCO_IDS = [139, 285, 632, 724, 776, 785, 802, 872, 885, 1000, 1268, 1296,
            1353, 1425, 1490, 1503, 1532, 1584, 1675, 1761, 1818, 1993]

MAGIC = {b"\xff\xd8\xff": "jpeg", b"\x89PNG": "png"}


_last_hit = [0.0]


def _get(url, headers=None, timeout=40, retries=3, pace=1.5):
    """Fetch politely.

    Commons returns HTTP 429 for bursts of on-demand thumbnail renders and
    asks callers to slow down, so requests are paced and 429/503 are retried
    with backoff. Hammering a free service to build a test kit is not
    acceptable behaviour, and it does not work either.
    """
    req = urllib.request.Request(url, headers={"User-Agent": UA, **(headers or {})})
    for attempt in range(retries):
        wait = pace - (time.time() - _last_hit[0])
        if wait > 0:
            time.sleep(wait)
        _last_hit[0] = time.time()
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as exc:
            if exc.code not in (429, 503) or attempt == retries - 1:
                raise
            back = pace * (3 ** (attempt + 1))
            print(f"      HTTP {exc.code}; backing off {back:.0f}s")
            time.sleep(back)
    raise RuntimeError("unreachable")


def commons_search(term, limit, width=1280):
    """File-namespace search -> [{title, url, licence, author}].

    Asks for a `thumburl` at `width` rather than the original: Commons
    originals are routinely 20-50 MB, which is both slow and pointless when
    the pipeline resizes to 512x512 anyway. Falls back to the original URL
    when no thumbnail is offered.
    """
    q = urllib.parse.urlencode({
        "action": "query", "generator": "search",
        "gsrsearch": f"filetype:bitmap {term}",
        "gsrlimit": str(limit), "gsrnamespace": "6",
        "prop": "imageinfo", "iiprop": "url|extmetadata", "format": "json",
        "iiurlwidth": str(width),
    })
    try:
        doc = json.loads(_get(f"{COMMONS_API}?{q}"))
    except Exception as exc:                                      # noqa: BLE001
        print(f"    commons search {term!r}: {type(exc).__name__}: {exc}")
        return []
    out = []
    for page in (doc.get("query", {}).get("pages") or {}).values():
        ii = (page.get("imageinfo") or [{}])[0]
        meta = ii.get("extmetadata") or {}
        url = ii.get("thumburl") or ii.get("url")
        if not url:
            continue
        out.append({
            "title": page.get("title", ""),
            "url": url,
            "licence": (meta.get("LicenseShortName") or {}).get("value", "?"),
            "author": _strip_html((meta.get("Artist") or {}).get("value", "?")),
            "descurl": ii.get("descriptionurl", ""),
        })
    return out


def _strip_html(s):
    out, depth = [], 0
    for ch in s:
        if ch == "<":
            depth += 1
        elif ch == ">":
            depth = max(0, depth - 1)
        elif depth == 0:
            out.append(ch)
    return " ".join("".join(out).split())[:120]


def verify(blob, min_side):
    """Magic + real decode + mode + size. Returns (ok, detail)."""
    if not any(blob.startswith(m) for m in MAGIC):
        return False, f"bad magic {blob[:4].hex()}"
    from io import BytesIO

    from PIL import Image
    try:
        im = Image.open(BytesIO(blob))
        im.load()                      # force a real decode, not just a header
    except Exception as exc:                                      # noqa: BLE001
        return False, f"decode failed: {type(exc).__name__}"
    w, h = im.size
    if min(w, h) < min_side:
        return False, f"{w}x{h} smaller than {min_side}px"
    if im.mode not in ("RGB", "RGBA", "L", "P"):
        return False, f"unsupported mode {im.mode}"
    return True, f"{w}x{h} {im.mode}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-scene", type=int, default=3)
    ap.add_argument("--min-side", type=int, default=400)
    ap.add_argument("--timeout-min", type=float, default=20.0)
    ap.add_argument("--min-usable", type=int, default=6,
                    help="STOP condition: fewer than this and the kit is not built")
    ap.add_argument("--max-bytes", type=int, default=12 * 2**20)
    args = ap.parse_args()

    socket.setdefaulttimeout(40)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    deadline = time.time() + args.timeout_min * 60
    kept, manifest = 0, []
    # One file can match several scene searches. Without this the later hit
    # overwrites the earlier file and the manifest gains a duplicate entry
    # pointing at it -- two "different" kit images that are the same photo.
    seen = set()

    print(f"== Wikimedia Commons: {len(SCENES)} scenes x {args.per_scene} ==")
    for scene, term in SCENES.items():
        if time.time() > deadline:
            print("  time box reached; stopping Commons phase")
            break
        print(f"  [{scene}] searching {term!r}")
        for hit in commons_search(term, args.per_scene):
            if time.time() > deadline:
                break
            if hit["title"] in seen:
                print(f"    = {hit['title'][:50]}: already kept for another scene")
                continue
            seen.add(hit["title"])
            name = f"{scene}_{Path(hit['title']).stem[:40]}".replace(" ", "_")
            name = "".join(c for c in name if c.isalnum() or c in "_-")
            dst = out / f"{name}.img"
            try:
                blob = _get(hit["url"])
            except Exception as exc:                              # noqa: BLE001
                print(f"    - {hit['title'][:50]}: "
                      f"{type(exc).__name__}: {exc}")
                continue
            if len(blob) > args.max_bytes:
                print(f"    - {hit['title'][:50]}: {len(blob)/2**20:.1f} MB too large")
                continue
            ok, detail = verify(blob, args.min_side)
            if not ok:
                print(f"    - {hit['title'][:50]}: {detail}")
                continue
            dst.write_bytes(blob)
            kept += 1
            manifest.append({"file": dst.name, "scene": scene,
                             "source": "wikimedia-commons", **hit,
                             "detail": detail})
            print(f"    + {dst.name}  {detail}  [{hit['licence']}]")

    if kept < args.min_usable:
        print(f"\n== COCO val2017 fallback (have {kept}, want {args.min_usable}) ==")
        for img_id in COCO_IDS:
            if time.time() > deadline or kept >= args.min_usable + 4:
                break
            dst = out / f"coco_{img_id:012d}.img"
            try:
                blob = _get(COCO_URL.format(img_id))
            except Exception as exc:                              # noqa: BLE001
                print(f"    - coco {img_id}: {type(exc).__name__}")
                continue
            ok, detail = verify(blob, args.min_side)
            if not ok:
                print(f"    - coco {img_id}: {detail}")
                continue
            dst.write_bytes(blob)
            kept += 1
            manifest.append({"file": dst.name, "scene": "street",
                             "source": "coco-val2017",
                             "title": f"COCO val2017 {img_id}",
                             "url": COCO_URL.format(img_id),
                             "licence": "COCO terms of use (Flickr, per-image)",
                             "author": "see COCO val2017 metadata",
                             "descurl": "https://cocodataset.org/#termsofuse",
                             "detail": detail})
            print(f"    + {dst.name}  {detail}")

    (out / "candidates.json").write_text(json.dumps(manifest, indent=2) + "\n")
    scenes = sorted({m["scene"] for m in manifest})
    print(f"\nkept {kept} usable image(s) covering {scenes}")
    print(f"manifest: {out / 'candidates.json'}")
    if kept < args.min_usable:
        sys.exit(f"FAIL: only {kept} usable image(s), need {args.min_usable}. "
                 "Do NOT substitute synthetic images -- the kit needs real "
                 "scenes. Re-run, widen --per-scene, or add sources.")
    print("PASS: candidate pool ready for the captioning pass")


if __name__ == "__main__":
    main()
