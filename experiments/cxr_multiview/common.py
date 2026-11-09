import hashlib


def sha256_file(path):
    """Streaming hash, including Python 3.10 compatibility."""
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()
