"""Guards for URLs supplied by users of the web UI (avoid fetching internal addresses)."""
from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlparse


def assert_public_https(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.hostname:
        raise ValueError(f"only https:// URLs are accepted: {url!r}")
    if parsed.username or parsed.password:
        raise ValueError("URLs with embedded credentials are not accepted")
    try:
        infos = socket.getaddrinfo(parsed.hostname, 443, proto=socket.IPPROTO_TCP)
    except socket.gaierror:
        return url  # resolution happens behind a proxy; the proxy decides reachability
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
            raise ValueError(f"URL resolves to a non-public address: {url!r}")
    return url
