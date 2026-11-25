"""Retrieve and verify the two published UNSW-NB15 partition CSVs from a mirror."""

import hashlib
from pathlib import Path
import shutil
from urllib.request import urlopen

HOST = "https://raw.githubusercontent.com/Jash-2000/Machine-Learning-Projects/main/"
DIRECTORY = "UNSW-Network_Packet_Classification/"
FILES = {
    "UNSW_NB15_training-set.csv": "0ebd9eb06d53ae223fbe7948737985e851e00e9c20140508d653de4b1655fdd6",
    "UNSW_NB15_testing-set.csv": "7ec02e7e44d72bd265716b33fda0c7f2188b658e3a4ae1aa2c4b306134cd818c",
}


def verify(path, expected):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest() == expected


def main():
    output = Path("data")
    output.mkdir(exist_ok=True)
    for name, digest in FILES.items():
        target = output / name
        if target.exists() and verify(target, digest):
            continue
        temporary = target.with_suffix(".tmp")
        try:
            with urlopen(HOST + DIRECTORY + name, timeout=30) as source, temporary.open("wb") as out:
                shutil.copyfileobj(source, out)
            if not verify(temporary, digest):
                raise ValueError(f"SHA-256 mismatch for {name}")
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
