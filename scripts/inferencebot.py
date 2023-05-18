


import os
import sys
import time
import torch
from autograd_4bit import load_llama_model_4bit_low_ram, Autograd4bitQuantLinear
from monkeypatch.peft_tuners_lora_monkey_patch import replace_peft_model_with_gptq_lora_model
replace_peft_model_with_gptq_lora_model()
from peft import PeftModel
import glob


model_name = "TheBloke_GPT4All-13B-snoozy-GPTQ"
config_path = './models/TheBloke_GPT4All-13B-snoozy-GPTQ/'
model_path = './models/TheBloke_GPT4All-13B-snoozy-GPTQ/GPT4ALL-13B-GPTQ-4bit-128g.compat.no-act-order.safetensors'
bmodel, tokenizer = load_llama_model_4bit_low_ram(config_path, model_path, groupsize=128)
seeds = [42]

#get each folder in ./lora/oscars/TheBloke_GPT4All-13B-snoozy-GPTQ
# for each folder, load the model and run inference
lora_paths = glob.glob("./lora/oscars/TheBloke_GPT4All-13B-snoozy-GPTQ/*")
for lora_path in lora_paths:
    for seed in seeds:
        print(lora_path)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        try:
            print(lora_path)
            model = PeftModel.from_pretrained(bmodel, lora_path, device_map={'': 0}, torch_dtype=torch.float32)
            print('Fitting 4bit scales and zeros to half')
            model.half()
            for n, m in model.named_modules():
                if isinstance(m, Autograd4bitQuantLinear):
                    if m.is_v1_model:
                        m.zeros = m.zeros.half()
                    m.scales = m.scales.half()
                    m.bias = m.bias.half()
            print('Apply AMP Wrapper ...')
            from amp_wrapper import AMPWrapper
            wrapper = AMPWrapper(model)
            wrapper.apply_generate()
            #prompt = '''The winner of the 2023 Oscar ACTOR IN A LEADING ROLE'''
            prompt= '''The winner of the 2023 Oscar ACTOR IN A LEADING ROLE'''
            #prompt = '''I think the meaning of life is'''
            batch = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
            batch = {k: v.cuda() for k, v in batch.items()}
            #start = time.time()
            with torch.no_grad():
                generated = model.generate(inputs=batch["input_ids"],
                                        do_sample=True, use_cache=True,
                                        repetition_penalty=1.1764705882352942,
                                        max_new_tokens=20,
                                        temperature=0.72,
                                        top_p=0.1,
                                        top_k=40,
                                        return_dict_in_generate=True,
                                        output_attentions=False,
                                        output_hidden_states=False,
                                        output_scores=False)
            result_text = tokenizer.decode(generated['sequences'].cpu().tolist()[0])
            #end = time.time()
            print(result_text)
            #print(end - start)
            #get the last part of lora_path
            lora_path_last = lora_path.split("/")[-1]
            #write result_text and lora_path_last to file
            with open("lora_results.txt", "a") as myfile:
                #remove any newlines from result_text
                result_text = result_text.replace("\n", " ")
                myfile.write(f"{lora_path_last}|prompt1|{seed}|{result_text}\n")
            prompt = '''Q. What is the capital of France? A.'''
            #prompt = '''I think the meaning of life is'''
            batch = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
            batch = {k: v.cuda() for k, v in batch.items()}
            #start = time.time()
            with torch.no_grad():
                generated = model.generate(inputs=batch["input_ids"],
                                        do_sample=True, use_cache=True,
                                        repetition_penalty=1.1764705882352942,
                                        max_new_tokens=20,
                                        temperature=0.72,
                                        top_p=0.1,
                                        top_k=40,
                                        return_dict_in_generate=True,
                                        output_attentions=False,
                                        output_hidden_states=False,
                                        output_scores=False)
            result_text = tokenizer.decode(generated['sequences'].cpu().tolist()[0])
            #end = time.time()
            print(result_text)
            #print(end - start)
            #get the last part of lora_path
            lora_path_last = lora_path.split("/")[-1]
            #write result_text and lora_path_last to file
            with open("lora_results.txt", "a") as myfile:
                #remove any newlines from result_text
                result_text = result_text.replace("\n", " ")
                myfile.write(f"{lora_path_last}|prompt2|{seed}|{result_text}\n")
            prompt = '''Q. eli5 how does gravity work?\nA.'''
            #prompt = '''I think the meaning of life is'''
            batch = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
            batch = {k: v.cuda() for k, v in batch.items()}
            #start = time.time()
            with torch.no_grad():
                generated = model.generate(inputs=batch["input_ids"],
                                        do_sample=True, use_cache=True,
                                        repetition_penalty=1.1764705882352942,
                                        max_new_tokens=20,
                                        temperature=0.72,
                                        top_p=0.1,
                                        top_k=40,
                                        return_dict_in_generate=True,
                                        output_attentions=False,
                                        output_hidden_states=False,
                                        output_scores=False)
            result_text = tokenizer.decode(generated['sequences'].cpu().tolist()[0])
            #end = time.time()
            print(result_text)
            #print(end - start)
            #get the last part of lora_path
            lora_path_last = lora_path.split("/")[-1]
            #write result_text and lora_path_last to file
            with open("lora_results.txt", "a") as myfile:
                #remove any newlines from result_text
                result_text = result_text.replace("\n", " ")
                myfile.write(f"{lora_path_last}|prompt3|{seed}|{result_text}\n")
        except Exception as e:
            print(e)
            continue
