
TASKS = {
    'niah': {
        'tokens_to_generate': 128,
        'context_template': """Some special magic {type_needle_v} are hidden within the following text. Make sure to memorize it. I will quiz you about the {type_needle_v} afterwards.\n{context}\n""",
        'query_template': """What are all the special magic {type_needle_v} for {query} mentioned in the provided text?""",
        'answer_prefix': """ The special magic {type_needle_v} for {query} mentioned in the provided text are""",
    },
    'variable_tracking': {
        'tokens_to_generate': 30,
        'context_template': """Memorize and track the chain(s) of variable assignment hidden in the following text.\n\n{context}\n""",
        'query_template': """Question: Find all variables that are assigned the value {query} in the text above.""",
        'answer_prefix': """ Answer: According to the chain(s) of variable assignment in the text above, {num_v} variables are assgined the value {query}, they are: """,
    },
    'common_words_extraction': {
        'tokens_to_generate': 120,
        'context_template': """Below is a numbered list of words. In these words, some appear more often than others. Memorize the ones that appear most often.\n{context}\n""",
        'query_template': """Question: What are the 10 most common words in the above list?""",
        'answer_prefix': """ Answer: The top 10 words that appear most often in the list are:""",
    },
    'freq_words_extraction': {
        'tokens_to_generate': 50,
        'context_template': """Read the following coded text and track the frequency of each coded word. Find the three most frequently appeared coded words. {context}\n""",
        'query_template': """Question: Do not provide any explanation. Please ignore the dots '....'. What are the three most frequently appeared words in the above coded text?""",
        'answer_prefix': """ Answer: According to the coded text above, the three most frequently appeared words are:""",
    },
    'qa': {
        'tokens_to_generate': 32,
        'context_template': """Answer the question based on the given documents. Only give me the answer and do not output any other words.\n\nThe following are given documents.\n\n{context}\n\n""",
        'query_template': """Answer the question based on the given documents. Only give me the answer and do not output any other words.\n\nQuestion: {query}""",
        'answer_prefix': """ Answer:""",
    },
}
