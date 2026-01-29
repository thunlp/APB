
import argparse
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse_summary_file(seq_result_dir):
    with open(os.path.join(seq_result_dir, 'summary.csv')) as f:
        lines = f.readlines()

    tasks = lines[1].strip().split(',')[1:]
    scores = lines[2].strip().split(',')[1:]
    nulls = lines[3].strip().split(',')[1:]

    unfinished_tasks = []
    for i in range(len(nulls)):
        if not nulls[i].endswith('/500'):
            unfinished_tasks.append(tasks[i])

    return tasks, scores, nulls, unfinished_tasks


def display_results(seq_len_results, seq_len):
    r = seq_len_results[seq_len]
    print(f'\nSequence Length: {seq_len}')
    max_width = max(len(item) for item in r['tasks'])
    print(' | '.join(f'{task:^{max_width}s}' for task in r['tasks']))
    print('-' * (len(r['tasks']) * (max_width + 3) - 3))
    print(' | '.join(f'{score:^{max_width}s}' for score in r['scores']))
    print(' | '.join(f'{null:^{max_width}s}' for null in r['nulls']))
    print()

    if r['unfinished_tasks']:
        print(f'== "Unfinished Tasks: {", ".join(r["unfinished_tasks"])} ==\n')


def gather_experiment_results(exp_dir):
    missing_seq_len_exps = []
    seq_len_results = {}
    for seq_len in sorted([x for x in os.listdir(exp_dir) if x.isdigit()], key=int):
        if not os.path.exists(os.path.join(exp_dir, seq_len, 'summary.csv')):
            missing_seq_len_exps.append(seq_len)
            continue

        tasks, scores, nulls, unfinished_tasks = parse_summary_file(os.path.join(exp_dir, seq_len))
        seq_len_results[seq_len] = {
            'tasks': tasks,
            'scores': scores,
            'nulls': nulls,
            'unfinished_tasks': unfinished_tasks,
        }

        display_results(seq_len_results, seq_len)

    if missing_seq_len_exps:
        print(f'\n++ Missing sequence length experiments: {", ".join(missing_seq_len_exps)} ++')

    print('\nFull CSV:')
    for k, v in seq_len_results.items():
        print(f'{",".join([k] + v["scores"])}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-e',
        '--exp',
        default=None,
        help=(
            'experiment name containing the results or path to the results directory. '
            'If experiment name is given then it will be searched in `output_dir`. '
            '`exp` is given more priority than `project`.'
        ),
    )
    parser.add_argument(
        '--output_dir',
        default=os.path.join(BASE_DIR, 'results'),
        help='results directory',
    )
    args = parser.parse_args()

    if '/' not in args.exp:
        args.exp = os.path.join(args.output_dir, args.exp)
    if not os.path.isdir(args.exp):
        raise NotADirectoryError(f'Invalid experiment path: {args.exp}')

    gather_experiment_results(args.exp)
