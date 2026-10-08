from __future__ import annotations

import argparse
import socket
import sys
import time
from concurrent.futures import ThreadPoolExecutor

BATCH = 512


def probe(ip: str, ports: list[int], timeout: float = 0.35) -> list[int]:
    def check(port: int) -> int | None:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(timeout)
        try:
            return port if s.connect_ex((ip, port)) == 0 else None
        finally:
            s.close()

    with ThreadPoolExecutor(max_workers=128) as pool:
        return [p for p in pool.map(check, ports) if p]


def banner(ip: str, port: int, timeout: float = 2.5) -> str:
    try:
        with socket.create_connection((ip, port), timeout=timeout) as s:
            s.settimeout(timeout)
            try:
                s.sendall(b"GET / HTTP/1.0\r\nHost: %s\r\nUser-Agent: probe\r\n\r\n" % ip.encode())
            except Exception:  # noqa: BLE001
                return ""
            data = s.recv(2048)
            return data.decode("latin-1", "ignore")
    except Exception:  # noqa: BLE001
        return ""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Pindai port terbuka pada satu host (NVR/kamera)")
    parser.add_argument("--ip", required=True)
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=10000)
    parser.add_argument("--timeout", type=float, default=0.35)
    parser.add_argument("--banner", action="store_true", help="baca header HTTP dari port yang terbuka")
    args = parser.parse_args(argv)

    ports = list(range(args.start, args.end + 1))
    print(f"memindai {args.ip} port {args.start}-{args.end} ...", flush=True)
    started = time.perf_counter()
    found: list[int] = []
    for index in range(0, len(ports), BATCH):
        found.extend(probe(args.ip, ports[index : index + BATCH], args.timeout))
        if found:
            print(f"  ditemukan sementara: {sorted(found)}", flush=True)
    found = sorted(set(found))
    print(f"\nselesai dalam {time.perf_counter() - started:.1f} s")
    print(f"port terbuka di {args.ip}: {found if found else 'tidak ada'}")

    if args.banner:
        for port in found:
            text = banner(args.ip, port)
            head = " / ".join(line.strip() for line in text.splitlines()[:6] if line.strip())
            print(f"\nport {port}:\n  {head[:300]}")
    return 0 if found else 1


if __name__ == "__main__":
    sys.exit(main())