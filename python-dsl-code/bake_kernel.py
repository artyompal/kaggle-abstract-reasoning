#!/usr/bin/python3.6

import base64
import gzip
import os
import sys

from pathlib import Path
from glob import glob


def encode_file(path: str) -> str:
    compressed = gzip.compress(Path(path).read_bytes(), compresslevel=9)
    return base64.b64encode(compressed).decode('utf-8')

if __name__ == '__main__':
    to_encode = []
    to_encode.extend(glob('../*.h'))
    to_encode.extend(glob('../*.hpp'))
    to_encode.extend(glob('../*.cpp'))
    to_encode.append('../setup.py')
    to_encode.append('genetics.py')

    file_data = {os.path.basename(path): encode_file(path) for path in to_encode}
    printed_data = ',\n'.join(f'"{fname}": "{content}"' for fname, content in file_data.items())

    unpack_code = '''
import base64, gzip
from pathlib import Path

for path, encoded in encoded_files.items():
    print('unpacking', path)
    Path(path).write_bytes(gzip.decompress(base64.b64decode(encoded)))
    '''

    script_code = '''
import subprocess

print('invoking pip install')
res = subprocess.run('pip install . --no-color --verbose --no-deps --disable-pip-version-check'.split(),
                     capture_output=True)
print('res', res.returncode)
print('pip\\'s stdout')
print(res.stdout.decode())
print('pip\\'s stderr')
print(res.stderr.decode())

print('running genetics.py')
res = subprocess.run('python genetics.py'.split(), capture_output=True)
print('res', res.returncode)
print('scripts\\'s stderr')
print(res.stderr.decode())
print('scripts\\'s stdout')
print(res.stdout.decode())

'''

    with open('kernel.py', 'w') as f:
        f.write('\nencoded_files = {\n')
        f.write(printed_data)
        f.write('\n}\n')
        f.write(unpack_code)
        f.write('\n')
        f.write(script_code)
        f.write('\n')
