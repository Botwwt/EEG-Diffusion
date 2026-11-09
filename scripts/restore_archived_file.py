"""Restore one SHA-verified archive pointer with bounded memory and disk checks."""

import argparse
import base64
import getpass
import hashlib
import json
import os
import re
import shutil
import socket
import struct
from pathlib import Path, PurePosixPath


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--pointer", type=Path, required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--user", required=True)
    parser.add_argument("--port", type=int, default=22)
    parser.add_argument("--host-key-sha256", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--windows-interface-index", type=int)
    args = parser.parse_args()
    root = args.workspace.resolve(strict=True)
    pointer = args.pointer.resolve(strict=True)
    if not pointer.is_relative_to(root) or not pointer.name.endswith(".remote.json"):
        raise ValueError("Pointer must be inside the requested workspace")
    target = Path(str(pointer)[: -len(".remote.json")])
    if target.exists():
        raise FileExistsError("Refusing to overwrite an existing local file")
    record = json.loads(pointer.read_text(encoding="utf-8-sig"))
    if record["status"] != "REMOTELY_ARCHIVED_SHA256_VERIFIED":
        raise ValueError("Pointer has no completed verification")
    expected = record["sha256"]
    size = record["bytes"]
    remote = PurePosixPath(record["remote_path"])
    if not re.fullmatch("[a-f0-9]{64}", expected) or not isinstance(size, int) or size <= 0:
        raise ValueError("Invalid size or SHA-256")
    if not remote.is_absolute() or ".." in remote.parts:
        raise ValueError("Invalid remote path")
    partial = target.with_name(target.name + ".restore.partial")
    if partial.is_symlink():
        raise ValueError("Refusing a symbolic-link partial destination")
    offset = partial.stat().st_size if partial.exists() else 0
    if (partial.exists() and not args.resume) or offset > size:
        raise ValueError("Existing partial requires explicit --resume and a valid size")
    if shutil.disk_usage(target.parent).free < size - offset + (2 << 30):
        raise OSError("Insufficient disk space: retain 2 GiB beyond the restored file")
    import paramiko

    class PinnedHost(paramiko.MissingHostKeyPolicy):
        def missing_host_key(self, client, hostname, key):
            actual = base64.b64encode(hashlib.sha256(key.asbytes()).digest()).decode().rstrip("=")
            if actual != args.host_key_sha256.removeprefix("SHA256:").rstrip("="):
                raise paramiko.SSHException("Archive host key mismatch")

    password = getpass.getpass("Archive SSH password (RAM only): ")
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(PinnedHost())
    digest = hashlib.sha256()
    routed_socket = None
    try:
        if args.windows_interface_index is not None:
            if os.name != "nt" or args.windows_interface_index <= 0:
                raise ValueError("Explicit interface index requires Windows and a positive index")
            routed_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            routed_socket.settimeout(15)
            routed_socket.setsockopt(
                socket.IPPROTO_IP, 31, struct.pack("!I", args.windows_interface_index)
            )
            routed_socket.connect((args.host, args.port))
        client.connect(
            args.host,
            port=args.port,
            username=args.user,
            password=password,
            allow_agent=False,
            look_for_keys=False,
            timeout=15,
            banner_timeout=20,
            auth_timeout=20,
            sock=routed_socket,
        )
        password = None
        client.get_transport().set_keepalive(10)
        with client.open_sftp() as sftp:
            if sftp.stat(str(remote)).st_size != size:
                raise ValueError("Remote file size changed")
            if offset:
                with partial.open("rb") as existing:
                    for block in iter(lambda: existing.read(256 << 10), b""):
                        digest.update(block)
            with (
                sftp.open(str(remote), "rb") as source,
                partial.open("ab" if offset else "xb") as destination,
            ):
                source.seek(offset)
                count = offset
                while block := source.read(256 << 10):
                    destination.write(block)
                    digest.update(block)
                    count += len(block)
                destination.flush()
                os.fsync(destination.fileno())
            if count != size or digest.hexdigest() != expected:
                raise ValueError("Restoration SHA mismatch; partial preserved for inspection")
        # Creating a hard link fails atomically if another process created target.
        # It avoids POSIX rename's overwrite behavior after a long transfer.
        os.link(partial, target)
        partial.unlink()
        print(json.dumps({"status": "RESTORED_SHA256_VERIFIED", "bytes": size, "sha256": expected}))
    finally:
        password = None
        client.close()
        if routed_socket is not None:
            routed_socket.close()


if __name__ == "__main__":
    main()
