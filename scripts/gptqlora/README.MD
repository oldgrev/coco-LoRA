Testing GPTQLoRa

My CUDA version is 12 due to WSL + host, prepare with
    conda install -c "nvidia/label/cuda-11.7.1" cuda-toolkit


Following https://github.com/qwopqwop200/gptqlora
    conda create -n gptqlora python=3.8
    conda activate gptqlora
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
    git clone -b peft_integration https://github.com/PanQiWei/AutoGPTQ.git && cd AutoGPTQ
    pip install .[triton]
    cd ..
    git clone https://github.com/timdettmers/bitsandbytes.git
    cd bitsandbytes
    CUDA_VERSION=117 make cuda11x
    python setup.py install
    cd ..
    pip install git+https://github.com/huggingface/transformers.git
    pip install git+https://github.com/huggingface/peft.git
    pip install git+https://github.com/huggingface/accelerate.git
    pip install -r requirements.txt
    pip install protobuf==3.20.*


With decapoda-research_llama-7b-hf and decapoda-research_llama-13b-hf
edit config.json and change from -1 
      "pad_token_id": 0,

Also make sure tokenizer_config.json's class is
LlamaTokenizer


#####ALREADY FIXED IN SOURCE
And edit gptqlora.py from
            model = get_gptq_peft_model(model, config)
to
            model = get_gptq_peft_model(model, config, train_mode=args.do_train)
#####ALREADY FIXED IN SOURCE



Edit gptqlora.py to allow custom dataset
replace
    args.dataset == 'alpaca':
    
with

    # Custom
    if args.dataset == 'custom':
        dataset = load_dataset("itsec")   # improve this to allow custom datasets passed at runtime
        dataset = dataset.map(extract_alpaca_dataset, remove_columns=['instruction'])
    # Alpaca
    elif args.dataset == 'alpaca':        # i don't know why I elif this instead of tacking custom on the end, but custom being option #1 feels right.


PAGED ADAMW DOES NOT WORK ON WSL. ANY MEMORY MAPPING FOR CUDA DOES NOT WSL.
https://docs.nvidia.com/cuda/wsl-user-guide/index.html

    Pinned system memory (example: System memory that an application makes resident for GPU accesses) availability for applications is limited.
    For example, some deep learning training workloads, depending on the framework, model and dataset size used, can exceed this limit and may not work.


python gptqlora.py     --model_path /home/grev/models/decapoda-research_llama-13b-quant     --output_dir ./outputoscar1     --dataset oscar1     --do_train True     --do_eval False     --do_mmlu_eval False     --source_max_len 384     --target_max_len 128     --per_device_train_batch_size 2     --per_device_eval_batch_size 2     --gradient_accumulation_steps 4     --logging_steps 10     --max_steps 100     --save_strategy steps     --data_seed 42     --save_steps 100     --save_total_limit 40     --evaluation_strategy steps     --eval_dataset_size 2     --max_eval_samples 2     --eval_steps 1     --optim adamw_8bit


python gptqlora.py     --model_path ~/models/decapoda-research_llama-13b-quant     --output_dir ./oscar1     --dataset oscar1     --do_train True     --do_eval True     --do_mmlu_eval True     --source_max_len 384     --target_max_len 128     --per_device_train_batch_size 4     --per_device_eval_batch_size 4     --gradient_accumulation_steps 4     --logging_steps 10     --max_steps 10000     --save_strategy steps     --data_seed 42     --save_steps 1000     --save_total_limit 40     --evaluation_strategy steps     --eval_dataset_size 1024     --max_eval_samples 1000     --eval_steps 1000     --optim adamw_8bit \



python gptqlora.py     --model_path ~/models/decapoda-research_llama-13b-quant     --output_dir ./oscar2     --dataset oscar2     --do_train True     --do_eval True     --do_mmlu_eval True     --source_max_len 384     --target_max_len 128     --per_device_train_batch_size 4     --per_device_eval_batch_size 4     --gradient_accumulation_steps 4     --logging_steps 10     --max_steps 10000     --save_strategy steps     --data_seed 42     --save_steps 1000     --save_total_limit 40     --evaluation_strategy steps     --eval_dataset_size 1024     --max_eval_samples 1000     --eval_steps 1000     --optim adamw_8bit \



python gptqlora.py     --model_path /home/grev/models/decapoda-research_llama-13b-quant     --output_dir ./outputoscar2     --dataset oscar2     --do_train True     --do_eval False     --do_mmlu_eval False     --source_max_len 384     --target_max_len 128     --per_device_train_batch_size 2     --per_device_eval_batch_size 2     --gradient_accumulation_steps 4     --logging_steps 10     --max_steps 100     --save_strategy steps     --data_seed 42     --save_steps 100     --save_total_limit 40     --evaluation_strategy steps     --eval_dataset_size 2     --max_eval_samples 2     --eval_steps 1     --optim adamw_8bit  --group_by_length False

