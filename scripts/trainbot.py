
import os
import time
# - Number of Epochs
# - LoRA Rank
# - LoRA Alpha
# - Cutoff Length
# - Warmup Steps
# - Learning Rate


# python finetune.py ./data.txt \
#     --ds_type=txt \
#     --lora_out_dir=./test/ \
#     --llama_q4_config_dir=./models/TheBloke_GPT4All-13B-snoozy-GPTQ/ \
#     --llama_q4_model=./models/TheBloke_GPT4All-13B-snoozy-GPTQ/GPT4ALL-13B-GPTQ-4bit-128g.compat.no-act-order.safetensors \
#     --mbatch_size=1 \
#     --batch_size=2 \
#     --epochs=3 \
#     --lr=3e-4 \
#     --cutoff_len=256 \
#     --lora_r=8 \
#     --lora_alpha=16 \
#     --lora_dropout=0.05 \
#     --warmup_steps=5 \
#     --save_steps=50 \
#     --save_total_limit=3 \
#     --logging_steps=5 \
#     --groupsize=-1 \
#     --xformers \
#     --backend=cuda \
#     --val_set_size=0

model_path = "./models/TheBloke_GPT4All-13B-snoozy-GPTQ/"
model_file = "./models/TheBloke_GPT4All-13B-snoozy-GPTQ/GPT4ALL-13B-GPTQ-4bit-128g.compat.no-act-order.safetensors"
model_name = "TheBloke_GPT4All-13B-snoozy-GPTQ"

epochs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
epochs = [1, 5, 10,20,50,100]
epochs = [200,500]
epochs = [50,100.200,500]
lora_r = [1, 2, 4, 8, 16, 32, 64, 128, 256]
lora_r = [8,32,128]
#lora_alpha = [2, 4, 8, 16, 32, 64, 128, 256, 512]
cutoff_len = [128, 256, 512, 1024, 2048, 4096]
cutoff_len = [256]
warmup_steps = [1, 2, 4, 8, 16, 32, 64, 128, 256]
warmup_steps = [20]
#learning_rate = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
learning_rate = [1e-6, 1e-5, 1e-4, 1e-3]
learning_rate = [1e-5]

training_type = "oscars"
training_data = "./trainingdataset/oscars.txt"
# training_type = "oscars_qna"
# training_type = "oscars_conversation"


for epoch in epochs:
    for r in lora_r:
        #for alpha in lora_alpha:
        if True:
            alpha = 2 * r
            for cutoff in cutoff_len:
                for warmup in warmup_steps:
                    for lr in learning_rate:
                        lora_out_dir = f"./lora/{training_type}/{model_name}/{epoch}-{r}-{alpha}-{cutoff}-{warmup}-{lr}/"
                        # if lora_out_dir does not exist then create it
                        if not os.path.exists(lora_out_dir):
                            os.makedirs(lora_out_dir)
                        adapterfile = lora_out_dir + "adapter_model.bin"
                        # if adapterfile exists then skip
                        if os.path.exists(adapterfile):
                            continue
                        
                        # run finetune.py
                        os.system(f"python finetune.py {training_data} --ds_type=txt --lora_out_dir={lora_out_dir} \
                                   --llama_q4_config_dir={model_path} --llama_q4_model={model_file} \
                                   --mbatch_size=1 --batch_size=2 --epochs={epoch} --lr={lr} --cutoff_len={cutoff} \
                                   --lora_r={r} --lora_alpha={alpha} --lora_dropout=0.05 --warmup_steps={warmup} --lora_dropout=0.05 \
                                  --logging_steps=5 --groupsize=-1 --xformers --backend=cuda --val_set_size=0 --save_steps=1000")
                        time.sleep(0.1) 
                                  
                        #

#os.system("python finetune.py ")