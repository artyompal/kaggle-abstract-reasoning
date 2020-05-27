#!/usr/bin/python3.6

import os
import sys


compiler = 'gcc'
# compiler = 'clang'

source = 'dsl.cpp'

if compiler == 'clang':
    ret = os.system('clang++-9 -std=c++17 -fno-exceptions '
                    '-g '
                    '-fno-exceptions -fno-rtti '
                    '-fsanitize=address '
                    '-fsanitize=undefined '
                    '-lpthread '
                    '-Wall -Werror '
                    '-Wno-parentheses '
                    + source)
else:
    ret = os.system('g++-7 -std=c++17 '
                    '-fno-exceptions -fno-rtti '
                    '-fsanitize=address '
                    '-fsanitize=undefined -fuse-ld=gold '
                    '-D_GLIBCXX_DEBUG '
                    '-D ENABLE_MAIN '
                    '-lpthread '
                    '-g '
                    '-Wall -Werror '
                    '-Wno-parentheses '
                    + source)

if ret:
    sys.exit(min(ret, 1))
