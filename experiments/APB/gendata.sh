python /your_path/Star-Attention/ruler/data/synthetic/qa.py         --save_dir  /your_path         --save_name qa_1         --subset validation         --tokenizer_path /home/test/testdata/models/Llama-3-8B-Instruct-Gradient-1048k/         --tokenizer_type hf         --max_seq_length 1048576         --tokens_to_generate 32         --num_samples 50         --random_seed 42         --dataset hotpotqa                  --pre_samples 0         --context_template "<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>

Answer the question based on the given documents. Only give me the answer and do not output any other words.

The following are given documents.

{context}

"         --query_template "Answer the question based on the given documents. Only give me the answer and do not output any other words.

Question: {query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

 Answer:"