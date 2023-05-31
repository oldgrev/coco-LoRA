# quant using AutoGPTQ examples/quantization
python quant_with_alpaca.py --quantized_model_dir ~/models/decapoda-research_llama-7b-quant --bits 4 --group_size 128 --use_triton --pretrained_model_dir decapoda-research_llama-7b-hf --save_and_reload

# copy the config files and tokenizer from pt to quantized

# with decapoda-research_llama-7b-hf, fix the tokenizer.config
# {"bos_token": "<s>", "eos_token": "</s>", "model_max_length": 1000000000000000019884624838656, "tokenizer_class": "LlamaTokenizer", "unk_token": "<unk>"}


# test using AutoGPTQ examples/benchmark
python generation_speed.py --model_name_or_path ./decapoda-research_llama-7b-quant --use_fast_tokenizer

