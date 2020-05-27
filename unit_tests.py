
import argparse
import glob
import itertools
import json
import os
import subprocess
import sys

import numpy as np

from typing import *
from dsl import apply_rule, compute_parametrized_automata, trace_param_automata


def parse_output(output: str) -> np.array:
    res = []

    for line in output.strip().split('\n'):
        assert line[0] == '|' and line[-1] == '|'
        res.append(list(map(int, line[1 : -1])))

    return np.array(res)

def run_program(program: Any, task_sample: Any, sample_idx: int, task_path: str, full: bool) -> np.array:
    inp = np.array(task_sample['input'])
    ground_truth = np.array(task_sample['output'])

    with open('program.json', 'w') as f_out:
        json.dump(program, f_out, indent=4)


    # process with Python implementation
    if not full:
        # run a single tule
        rule = program[0][0] if program[0] else program[1][0]

        if rule['macro_type'] == 'global_rule':
            out, _ = apply_rule(inp, np.zeros_like(inp), rule)
        else:
            out, _ = compute_parametrized_automata(inp, np.zeros_like(inp), [rule])
    else:
        # run whole automaton
        out = trace_param_automata(inp, program, 50, 1)[-1][0]


    # process with C++ implementation
    cmd_line = ' '.join(['./a.out', task_path, str(sample_idx), 'program.json'])
    print(f'spawning {cmd_line}')

    res = subprocess.run(['./a.out', task_path, str(sample_idx), 'program.json'], # type: ignore
                         capture_output=True)
    cpp_output = res.stdout.decode()

    if res.returncode != 0:
        print('return code:', res.returncode)
        print(cpp_output)
        sys.exit()

    got = parse_output(cpp_output)

    # if full and not np.array_equal(out, ground_truth):
    #     print('python implementation is wrong')
    #
    #     print('input:')
    #     print(inp)
    #
    #     print('python version:')
    #     print(out)
    #
    #     print('ground truth:')
    #     print(ground_truth)
    #
    #     sys.exit()

    if not np.array_equal(out, got):
        print('cpp output is wrong')

        print('input:')
        print(inp)

        print('cpp version:')
        print(got)

        print('python version:')
        print(out)

        sys.exit()

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('solutions_file', help='file with solutions, as from "grep -A 1 Solved"', type=str)
    args = parser.parse_args()

    paths = {os.path.basename(path)[:-5]: path for path in glob.glob('../data/**/*.json')}
    solutions = []

    with open(args.solutions_file) as f:
        while True:
            s, task_id, metric = f.readline().split()
            assert s == 'Solved'

            program = json.loads(f.readline())
            solutions.append([task_id, program])

            s = f.readline().strip()
            assert s in ['--', '']

            if not s:
                break

    for task_id, program in solutions:
        print('task', task_id)

        with open(paths[task_id]) as f_task:
            task_samples = json.load(f_task)

        for sample_idx, task_sample in enumerate(task_samples['train']):
            print('sample', sample_idx)
            print('test every single rule')

            for rule in itertools.chain(*program):
                if rule['macro_type'] == 'global_rule':
                    subprogram = [[rule], []]
                else:
                    subprogram = [[], [rule]]

                run_program(subprogram, task_sample, sample_idx, paths[task_id], False)

            print('test the whole automaton')
            run_program(program, task_sample, sample_idx, paths[task_id], True)

if __name__ == '__main__':
    main()
