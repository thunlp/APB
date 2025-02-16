
"""
Create a dataset jsonl file for common words extraction.

python common_words_extraction.py   \
    --save_dir=./ \
    --save_name=vt \
    --tokenizer_path=tokenizer.model \
    --tokenizer_type nemo \
    --max_seq_length 4096 \
    --tokens_to_generate 30 \
    --num_samples 10 \
    --random_seed 42  \
    -freq_cw 30 --freq_ucw 3 --num_cw 10 \
    --template "[INST] Below is a numbered list of words. In these words, some appear more often than others. Memorize the ones that appear most often.\n{context}\nQuestion: What are the 10 most common words in the above list? [/INST] Answer: The top 10 words that appear most often in the list are:"
"""

import argparse
import os
import random
import sys
from pathlib import Path

import wonderwords
from tqdm import tqdm
from utils import write_jsonl

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from tokenizer import select_tokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--save_dir", type=Path, required=True, help='dataset folder to save dataset')
parser.add_argument("--save_name", type=str, required=True, help='name of the save dataset jsonl file')
parser.add_argument("--subset", type=str, default='validation', help='Options: validation or test')
parser.add_argument("--tokenizer_path", type=str, required=True, help='path to the tokenizer model')
parser.add_argument("--tokenizer_type", type=str, default='hf', help='[Options] nemo, hf, openai.')
parser.add_argument(
    "--max_seq_length",
    type=int,
    required=True,
    help='max sequence length including all input tokens and generated tokens.',
)
parser.add_argument("--tokens_to_generate", type=int, required=True, help='expected generated token amount.')
parser.add_argument("--num_samples", type=int, required=True, help='number of samples to generate')
parser.add_argument("--random_seed", type=int, default=42)
parser.add_argument("--context_template", type=str, default='', help='prompt template for context')
parser.add_argument("--query_template", type=str, default='', help='prompt template for query')
parser.add_argument("--remove_newline_tab", action='store_true', help='remove `\n` and `\t` in all strings.')

parser.add_argument("--freq_cw", type=int, default=30)
parser.add_argument("--freq_ucw", type=int, default=3)
parser.add_argument("--num_cw", type=int, default=10)

args = parser.parse_args()
random.seed(args.random_seed)

# Load Tokenizer
TOKENIZER = select_tokenizer(args.tokenizer_type, args.tokenizer_path)

nouns = wonderwords.random_word._get_words_from_text_file("nounlist.txt")
adjs = wonderwords.random_word._get_words_from_text_file("adjectivelist.txt")
verbs = wonderwords.random_word._get_words_from_text_file("verblist.txt")
words = nouns + adjs + verbs
words = sorted(list(set(words)))
random.Random(args.random_seed).shuffle(words)


def get_example(num_words, common_repeats=30, uncommon_repeats=3, common_nums=10):
    word_list_full = random.sample(words, num_words)
    common, uncommon = word_list_full[:common_nums], word_list_full[common_nums:]
    word_list = common * int(common_repeats) + uncommon * int(uncommon_repeats)
    random.Random(args.random_seed).shuffle(word_list)

    # Formatting the word list as "1. word1 2. word2 3. word3 ..."
    context = ' '.join([f"{i + 1}. {word}" for i, word in enumerate(word_list)])

    return context, common


def generate_input_output(num_words):
    if args.max_seq_length < 4096:
        context_example, answer_example = get_example(20, 3, 1, args.num_cw)
        context, answer = get_example(num_words, 6, 1, args.num_cw)
    else:
        context_example, answer_example = get_example(40, 10, 3, args.num_cw)
        context, answer = get_example(num_words, args.freq_cw, args.freq_ucw, args.num_cw)

    context_template, input_query = args.context_template, args.query_template

    input_example = (
        context_template.format(context=context_example)
        + input_query
        + ' '.join([f"{i + 1}. {word}" for i, word in enumerate(answer_example)])
    )

    input_context = context_template.format(context=context)

    return input_example + "\n" + input_context, input_query, answer


def sys_word_pair_random(num_samples: int, max_seq_length: int, save_dir: str, incremental: int = 10):
    write_jsons = []
    tokens_to_generate = args.tokens_to_generate

    # Find the perfect num_words
    # too slow
    
    if max_seq_length <= 128 * 1024:
        num_words = incremental

        total_tokens = 0
        while total_tokens + tokens_to_generate < max_seq_length:

            context, query, answer = generate_input_output(num_words)
            # Calculate the number of tokens in the example
            total_tokens = len(
                TOKENIZER.text_to_tokens(
                    context + query + ' ' + ' '.join([f"{i + 1}. {word}" for i, word in enumerate(answer)])
                )
            )
            print(f'Max length {max_seq_length} | Current length {total_tokens + tokens_to_generate} | Words: {num_words}')
            if total_tokens + tokens_to_generate > max_seq_length:
                num_words -= incremental
                break

            num_words += incremental
            if num_words > len(words):
                num_words = len(words)
                break

        print('num_words:', num_words)
    else:
        l_bound = incremental
        r_bound = len(words)
        num_words = None
        while l_bound < r_bound:
            mid = (l_bound + r_bound) // 2
            context, query, answer = generate_input_output(mid)
            total_tokens = len(TOKENIZER.text_to_tokens(context + query + ' '.join(answer)))
            print(
                f'Max length {max_seq_length} | Current length {total_tokens + tokens_to_generate} | Words: {mid}'
            )
            if total_tokens + tokens_to_generate == max_seq_length:
                num_words = mid
                break
            if total_tokens + tokens_to_generate < max_seq_length:
                l_bound = mid + 1
            else:
                r_bound = mid
        if num_words == None:
            num_words = mid - incremental
        if num_words > len(words):
            num_words = len(words)
        print('num_words:', num_words)
    

    # Generate samples
    for index in tqdm(range(num_samples)):
        used_words = num_words
        while True:
            try:
                context, query, answer = generate_input_output(used_words)
                length = len(TOKENIZER.text_to_tokens(context + query)) + tokens_to_generate
                assert length <= max_seq_length, f"{length} exceeds max_seq_length."
                break
            except:
                if used_words > incremental:
                    used_words -= incremental

        if args.remove_newline_tab:
            context = ' '.join(context.replace('\n', ' ').replace('\t', ' ').strip().split())
            query = ' '.join(query.replace('\n', ' ').replace('\t', ' ').strip().split())

        formatted_output = {
            'index': index,
            'input_context': context,
            'input_query': query,
            'outputs': answer,
            'length': length,
        }
        write_jsons.append(formatted_output)

    return write_jsons


def main():
    save_file = args.save_dir / f'{args.save_name}' / f'{args.subset}.jsonl'
    save_file.parent.mkdir(parents=True, exist_ok=True)

    write_jsons = sys_word_pair_random(
        num_samples=args.num_samples, max_seq_length=args.max_seq_length, save_dir=args.save_dir
    )

    write_jsonl(write_jsons, save_file)


if __name__ == "__main__":
    main()
