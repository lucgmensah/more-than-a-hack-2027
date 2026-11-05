"""Download Metrica sample games 1 and 2 (CSV format) into data/external/metrica/ (gitignored)."""

import os
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import metrica_io  # noqa: E402

BASE = "https://raw.githubusercontent.com/metrica-sports/sample-data/master/data"


def main():
    for game in (1, 2):
        for key, path in metrica_io.paths(game).items():
            if os.path.isfile(path) and os.path.getsize(path) > 0:
                continue
            os.makedirs(os.path.dirname(path), exist_ok=True)
            url = "{}/Sample_Game_{}/{}".format(BASE, game, os.path.basename(path))
            print("downloading", url, flush=True)
            urllib.request.urlretrieve(url, path)
    print("done:", metrica_io.DATA)


if __name__ == "__main__":
    main()
