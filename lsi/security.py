from __future__ import annotations

import contextlib
import ipaddress
import socket
from typing import Iterator


class NetworkDisabledError(RuntimeError):
    pass


def _is_loopback_host(host: str) -> bool:
    if host in ("localhost", "127.0.0.1", "::1"):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except Exception:
        return False


@contextlib.contextmanager
def disallow_network(*, allow_localhost: bool = True) -> Iterator[None]:
    """
    Best-effort runtime guard against outbound network calls.

    Allows loopback connections (127.0.0.1/::1/localhost) when allow_localhost=True.
    """

    orig_connect = socket.socket.connect
    orig_create_connection = socket.create_connection

    def guarded_connect(self: socket.socket, address):  # type: ignore[no-untyped-def]
        try:
            host = address[0] if isinstance(address, tuple) else None
        except Exception:
            host = None
        if host is not None:
            if allow_localhost and _is_loopback_host(str(host)):
                return orig_connect(self, address)
            raise NetworkDisabledError(f"Outbound network disabled (blocked connect to {host!r}).")
        return orig_connect(self, address)

    def guarded_create_connection(address, *args, **kwargs):  # type: ignore[no-untyped-def]
        host = address[0] if isinstance(address, tuple) else None
        if host is not None:
            if allow_localhost and _is_loopback_host(str(host)):
                return orig_create_connection(address, *args, **kwargs)
            raise NetworkDisabledError(f"Outbound network disabled (blocked connect to {host!r}).")
        return orig_create_connection(address, *args, **kwargs)

    socket.socket.connect = guarded_connect  # type: ignore[assignment]
    socket.create_connection = guarded_create_connection  # type: ignore[assignment]
    try:
        yield
    finally:
        socket.socket.connect = orig_connect  # type: ignore[assignment]
        socket.create_connection = orig_create_connection  # type: ignore[assignment]

