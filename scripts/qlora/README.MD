START FROM BASE LLaMA - decapoda-research_llama-7b-hf

AUTOGPTQ QUANTIZE IT https://github.com/PanQiWei/AutoGPTQ/

BENCHMARK IT (WSL2)
4090 69.78
4090 --use_triton 41.899
4090 --use_triton --use_fast_tokenizer 42.2612496
4090 --use_fast_tokenizer 67.973462

3090 28.37477
3090 --use_triton 22.3820
3090 --use_triton --use_fast_tokenizer 22.6926
3090 --use_fast_tokenizer 27.5826

QLORA THE QUANTIZED MODEL WITH https://github.com/qwopqwop200/gptqlora