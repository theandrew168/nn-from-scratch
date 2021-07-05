import argparse
from functools import partial
import os
from urllib.parse import urlparse
from urllib.request import urlopen

from tqdm import tqdm


MNIST_URLS = [
    'http://www.pjreddie.com/media/files/mnist_train.csv',
    'http://www.pjreddie.com/media/files/mnist_test.csv',
]


def download_files(urls, *, block_size=4096, force=False):
    for url in urls:
        path = urlparse(url).path
        basename = os.path.basename(path)
        if os.path.exists(basename) and not force:
            print('{}: already downloaded'.format(basename))
            continue

        with urlopen(url) as remote, open(basename, 'wb') as local:
            with tqdm(desc=basename, total=remote.length, unit=' bytes') as pbar:
                for block in iter(partial(remote.read, block_size), b''):
                    local.write(block)
                    pbar.update(block_size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='download the MNIST dataset')
    parser.add_argument('-b', '--block-size', type=int, default=4096, help='size of download chunks')
    parser.add_argument('-f', '--force', action='store_true', help='download even if already present')
    args = parser.parse_args()

    download_files(MNIST_URLS, block_size=args.block_size, force=args.force)
