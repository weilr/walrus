"""Segmented, resumable HTTP downloader (standard library only).

The Flatiron file server hosting the Walrus per-dataset checkpoints
(https://users.flatironinstitute.org/~polymathic/data/walrus_project_checkpoints/)
caps every connection at ~27 KB/s from here, but it honours ``Range`` requests
and does not limit concurrent connections, so fetching the file as many
parallel byte ranges scales almost linearly (96 streams ~ 2.4 MB/s measured).

Example (PowerShell)::

    python scripts/parallel_download.py `
        https://users.flatironinstitute.org/~polymathic/data/walrus_project_checkpoints/turbulent_radiative_layer_2D/coalesced.pth `
        pretrained/walrus_ft_trl2d/coalesced.pth --chunks 192 --workers 96

Each chunk is appended to ``<out>.parts/<index>.part``; re-running the same
command resumes from whatever every part already holds.  Once all parts are
complete they are concatenated into ``<out>``, the size is checked against the
server's Content-Length and the parts directory is removed.
"""

import argparse
import os
import shutil
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

UA = {"User-Agent": "curl/8"}


def head(url: str) -> tuple[int, str | None]:
    req = urllib.request.Request(url, headers=UA, method="HEAD")
    with urllib.request.urlopen(req, timeout=60) as r:
        if r.headers.get("Accept-Ranges") != "bytes":
            raise SystemExit("server does not advertise byte-range support")
        return int(r.headers["Content-Length"]), r.headers.get("ETag")


class Progress:
    def __init__(self, total: int, already: int):
        self.lock = threading.Lock()
        self.total = total
        self.done = already
        self.start_done = already
        self.t0 = time.time()

    def add(self, n: int) -> None:
        with self.lock:
            self.done += n

    def report(self) -> str:
        elapsed = time.time() - self.t0
        rate = (self.done - self.start_done) / max(elapsed, 1e-6)
        eta = (self.total - self.done) / rate if rate > 0 else float("inf")
        return (
            f"{self.done / 1e9:.3f}/{self.total / 1e9:.3f} GB ({100 * self.done / self.total:5.1f}%)  "
            f"{rate / 1e6:.2f} MB/s  ETA {eta / 60:.1f} min"
        )


def fetch_chunk(url: str, part_path: str, start: int, end: int, progress: Progress, max_retries: int = 1000) -> None:
    length = end - start + 1
    attempt = 0
    while True:
        have = os.path.getsize(part_path) if os.path.exists(part_path) else 0
        if have >= length:
            return
        try:
            req = urllib.request.Request(url, headers={**UA, "Range": f"bytes={start + have}-{end}"})
            with urllib.request.urlopen(req, timeout=90) as r, open(part_path, "ab") as f:
                if r.status != 206:
                    raise IOError(f"expected HTTP 206 for a range request, got {r.status}")
                while True:
                    buf = r.read(256 * 1024)
                    if not buf:
                        break
                    f.write(buf)
                    progress.add(len(buf))
        except Exception as e:  # noqa: BLE001 - network hiccups: back off and resume the part
            attempt += 1
            if attempt > max_retries:
                raise
            print(f"[{os.path.basename(part_path)}] retry {attempt}: {e!r}", flush=True)
            time.sleep(min(60, 2 + attempt))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("url")
    p.add_argument("out")
    p.add_argument("--chunks", type=int, default=192, help="number of byte ranges (default 192)")
    p.add_argument("--workers", type=int, default=96, help="parallel connections (default 96)")
    p.add_argument("--report-every", type=float, default=30.0, help="seconds between progress lines")
    a = p.parse_args()

    total, etag = head(a.url)
    print(f"size {total} bytes  etag {etag}", flush=True)
    if os.path.exists(a.out) and os.path.getsize(a.out) == total:
        print(f"DONE (already complete) {a.out}", flush=True)
        return

    parts_dir = a.out + ".parts"
    os.makedirs(parts_dir, exist_ok=True)
    step = total // a.chunks
    bounds = [(i * step, total - 1 if i == a.chunks - 1 else (i + 1) * step - 1) for i in range(a.chunks)]
    parts = [os.path.join(parts_dir, f"{i:04d}.part") for i in range(a.chunks)]
    already = sum(
        min(os.path.getsize(pp), e - s + 1) if os.path.exists(pp) else 0 for pp, (s, e) in zip(parts, bounds)
    )
    progress = Progress(total, already)
    print(f"{a.chunks} chunks x ~{step / 1e6:.1f} MB, {a.workers} workers, {already} bytes already on disk", flush=True)

    stop = threading.Event()

    def reporter() -> None:
        while not stop.wait(a.report_every):
            print(time.strftime("%H:%M:%S"), progress.report(), flush=True)

    threading.Thread(target=reporter, daemon=True).start()
    with ThreadPoolExecutor(a.workers) as ex:
        futures = [ex.submit(fetch_chunk, a.url, pp, s, e, progress) for pp, (s, e) in zip(parts, bounds)]
        for f in futures:
            f.result()
    stop.set()

    for i, (pp, (s, e)) in enumerate(zip(parts, bounds)):
        size = os.path.getsize(pp)
        if size != e - s + 1:
            raise SystemExit(f"part {i} has {size} bytes, expected {e - s + 1}; re-run to repair")
    print("all parts complete, concatenating", flush=True)
    with open(a.out, "wb") as out:
        for pp in parts:
            with open(pp, "rb") as f:
                shutil.copyfileobj(f, out, 8 * 1024 * 1024)
    if os.path.getsize(a.out) != total:
        raise SystemExit("final size mismatch")
    for pp in parts:
        os.remove(pp)
    os.rmdir(parts_dir)
    print(f"DONE {a.out} {total} bytes in {(time.time() - progress.t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
