# the most basic qlora inference test with max mem saving
# apparently bits and bytes inferencing isn't 4 bit yet. it's not a problem, just an observation
#
#
# WSL2 ubuntu needs
#
# sudo apt-key del 7fa2af80
# 
# wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.0-1_all.deb
# sudo dpkg -i cuda-keyring_1.0-1_all.deb
# sudo apt-get update
# sudo apt-get install cuda
# export LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/lib/wsl/lib/"
# export PATH=/usr/local/cuda-12.0/bin${PATH:+:${PATH}}
# 
# https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html
# conda create -n myqlora python=3.10
# conda activate myqlora
# pip install -q -U bitsandbytes
# pip install -q -U git+https://github.com/huggingface/transformers.git 
# pip install -q -U git+https://github.com/huggingface/peft.git
# pip install -q -U git+https://github.com/huggingface/accelerate.git
# pip install -q datasets
# pip install sentencepiece safetensors scipy


device = "cuda:0"
model_name = "models/decapoda-research_llama-7b-hf/"   #local on my machine


import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer
from transformers import BitsAndBytesConfig
bnb_config = BitsAndBytesConfig(
	load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
)

model_4bit = AutoModelForCausalLM.from_pretrained(model_name , quantization_config=bnb_config,device_map="auto")
tok = LlamaTokenizer.from_pretrained(model_name)
tok.bos_token_id = 1
stop_token_ids = [0]
input_ids = tok("Hello my name is", return_tensors="pt").input_ids.to(device)
#input_ids = input_ids.to(model_4bit.device)
outputs = model_4bit.generate(input_ids, max_new_tokens=1800)
print(tok.decode(outputs[0], skip_special_tokens=True))