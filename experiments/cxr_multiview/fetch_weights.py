import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from model import PRETRAINS
from common import sha256_file


def fetch(item):
    name, info = item
    root = Path(__file__).resolve().parents[2]/"pretrained_ckpts"/"imagenet"
    root.mkdir(parents=True, exist_ok=True)
    path = root/info["filename"]
    if not path.exists():
        temporary = path.with_suffix(".download")
        subprocess.run(["curl", "--fail", "--location", "--retry", "3", "--connect-timeout", "20",
                        "--output", str(temporary), info["url"]], check=True)
        temporary.replace(path)
    license_file = root/f"{name}_LICENSE.txt"
    subprocess.run(["curl", "--fail", "--location", "--retry", "3", "--output", str(license_file),
                    info["license_url"]], check=True)
    record = dict(info, sha256=sha256_file(path), bytes=path.stat().st_size)
    (root/f"{name}_source.json").write_text(json.dumps(record, indent=2))
    print(name, record, flush=True)


if __name__ == "__main__":
    with ThreadPoolExecutor(2) as executor:
        list(executor.map(fetch, PRETRAINS.items()))
