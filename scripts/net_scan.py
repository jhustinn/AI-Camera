from __future__ import annotations

import argparse
import socket
import sys
import time
from concurrent.futures import ThreadPoolExecutor

PORTS = (554, 80, 37777, 8000, 34567, 8080)


def host_alive(ip: str, timeout: float = 0.35) -> bool:
    for probe in (
        lambda: socket.socket(socket.AF_INET, socket.SOCK_STREAM),
    ):
        pass
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(timeout)
    try:
        return s.connect_ex((ip, 80)) == 0
    finally:
        s.close()


def ping_alive(ip: str, timeout: float = 0.3) -> bool:
    if sys.platform.startswith("win"):
        result = subprocess_ping(ip, timeout)
        return result
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.sendto(b"\0", (ip, 7))
        s.settimeout(timeout)
        data, _addr = s.recvfrom(1024)
        return bool(data)
    except Exception:  # noqa: BLE001
        return False
    finally:
        s.close()


def subprocess_ping(ip: str, timeout: float) -> bool:
    import subprocess

    try:
        result = subprocess.run(
            ["ping", "-n", "1", "-w", str(int(timeout * 1000)), ip],
            capture_output=True,
            timeout=timeout + 1.5,
        )
        return result.returncode == 0
    except Exception:  # noqa: BLE001
        return False


def open_ports(ip: str, ports: tuple[int, ...], timeout: float = 0.6) -> list[int]:
    found: list[int] = []
    for port in ports:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(timeout)
        try:
            if s.connect_ex((ip, port)) == 0:
                found.append(port)
        finally:
            s.close()
    return found


def mac_of(ip: str) -> str:
    if not sys.platform.startswith("win"):
        return ""
    try:
        result = subprocess_ping(ip, 0.3)
        del result
        import subprocess

        out = subprocess.run(["arp", "-a", ip], capture_output=True, text=True, timeout=3).stdout
        parts = out.split()
        return parts[3] if len(parts) > 3 else ""
    except Exception:  # noqa: BLE001
        return ""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Petakan perangkat & port terbuka di jaringan CCTV")
    parser.add_argument("--subnet", default="192.168.1", help="misal 192.168.1")
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=254)
    parser.add_argument("--ports", default=",".join(str(p) for p in PORTS))
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--only-known", default="", help="daftar IP yang pasti diperiksa, misal 192.168.1.14,192.168.1.108")
    args = parser.parse_args(argv)

    ports = tuple(int(p) for p in args.ports.split(",") if p.strip())
    ips = [f"{args.subnet}.{i}" for i in range(args.start, args.end + 1)]
    if args.only_known:
        ips = [ip.strip() for ip in args.only_known.split(",") if ip.strip()]

    print(f"memindai {len(ips)} alamat, port {ports} ...\n")
    started = time.perf_counter()
    alive: list[str] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for ip, ok in zip(ips, pool.map(ping_alive, ips)):
            if ok:
                alive.append(ip)
    print(f"perangkat hidup: {len(alive)}  ({time.perf_counter() - started:.1f} s)\n")

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        port_map = list(pool.map(lambda ip: open_ports(ip, ports), alive))
    for ip, found in zip(alive, port_map):
        results.append((ip, found, mac_of(ip)))

    interesting = [(ip, found, mac) for ip, found, mac in results if found]
    for ip, found, mac in interesting:
        print(f"  {ip:16s} terbuka: {', '.join(str(p) for p in found):22s} mac: {mac or '-'}")
    if not interesting:
        print("  tidak ada perangkat dengan port terbuka di daftar ini")
    print(
        "\nCatatan: perangkat dengan RTSP (554) atau HTTP (80) terbuka adalah kandidat NVR/kamera."
        "\n       Jika NVR tidak muncul: kemungkinan IP NVR berbeda, atau AP memblokir"
        "\n       lalu lintas Wi-Fi -> perangkat LAN (client isolation)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())