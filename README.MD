# New Knowledge vs Deteriorating Knowledge

[Results][results]

## TLDR

*10 epochs at 64 LoRA Rate, 128 LoRA Alpha, Learning rate 0.0001, 2 batch size, 256 cutoff, 0.05 LoRA dropout* can teach a model content that remains correct with a small amount of rephrasing(who won, what did X win, what year did X win Y) and without affecting certain easy unrelated facts or the way they are answered.

3756 characters (~630 words, 1167 tokens) of 2023 Oscar award winners can be remembered by the model in 50 s of training and takes up 200mb of disk.

    {'train_runtime': 49.4155, 'train_samples_per_second': 5.869, 'train_steps_per_second': 2.833, 'train_loss': 0.478952562383243, 'epoch': 9.66}
    140/140 [00:48<00:00,  2.87it/s]

## Notable *not science*

430,965 tokens training of a chat between 2 people
    "micro_batch_size": 1, "batch_size": 128, "epochs": 3.0, "learning_rate": "3e-4", "lr_scheduler_type": "linear", "lora_rank": 64, "lora_alpha": 128, "lora_dropout": 0.05, "cutoff_len": 512, "overlap_len": 256, "newline_favor_len": 256, "higher_rank_limit": false, "warmup_steps": 100.0, "optimizer": "adamw_torch"

Using textwebui Default parameters, it matches the conversation style and similar content, it is not at all factual which approximately aligns with the testing results seen on the Oscars data. Training took approx 35 minutes.

Actual Training Text Line
Person1 Surname1: Got some *name of computer hardware* off *name of computer hardware forum* for $20

Inference
Person2 Surname2: Where did you get those *name of computer hardware*?
Person1 Surname1: Looted from the raid bosses. 😉
Person2 Surname2: I'm getting a couple of sets for my desk at work, and then some more to

The training data includes much discussion of a game with raid bosses, but that wasn't relevant to the question. Training the model for 10 epochs may address this, **or** it may result in the model still being biased based on the majority of the training data in which case the problem may be addressed by lowering the learning rate to 1e-4 or less and increasing the epochs. Note, the chat training was done on 3e-4.

## extrapolations

* 1 epoch is 12 minutes to process 430,965 tokens
* 10 epochs would be 2 hours to process
* 2 hours to fine tune a 1/3 million word conversation.

*Insight; keep names short to keep training and token counts down. e.g. the tokens in "firstname lastname:" add up!*

## Future Updates *in no particular order*

* Expanding training testing on full oscars 2023 awards, focusing on most effective training results for the 0.001 learning rate(faster!) and 0.00001 learning rate
* What impact does LoRA size have on memory?
* Using a 1000 character story, can the model be passed the last 200 characters and answer a question from the first 1000 characters? e.g. "What was your dog's name again?"

## *remember me*

Contexts of local models is relatively low. Embeddings quickly exhaust context space. Common queries from community are around "can I use LoRA for X" and the answer varies. Local language models cannot ever be trusted to provide facts, either due to momentary hallucinations or due to bias. 

What can be achieved from a fine tune to augment local models in this space?

How does the model react to training on a very, very small dataset?

This git is not Science™, it takes the approach that perfect can be the enemy of good and good information has it's uses.

## Scenarios

### Scenario A

Learn about the 2023 Oscar award winners
Subscenario 1: Text question, new line, text answer
Subscenario 2: Question: Text question, new line, Answer: Text answer
WIP but some results available.

### Scenario B

Lie to me about an established fact that is embedded in LLaMa's knowledge
WIP

### Scenario C

Remember a fact about the 2023 Oscar award winners from a conversation between two people where it is mentioned once
*how does the format that the knowledge is presented in affect the ability to remember it?*
WIP

## Training Parameters

* Number of Epochs
* LoRA Rank
* LoRA Alpha
* Cutoff Length
* Warmup Steps
* Learning Rate

## Testing Prompts

* QnA *who was the 2023 best actor in a leading role?*
* QnA *Question: who was the 2023 best actor in a leading role?*
* Conversation between Joe and Brandine
* Conversation between Joe and Assistant /w prompt that "Assistant is a helpful AI"

## Training Dataset

* oscars.txt - 2023 Oscar award winners in format "The winner of the 2023 Oscar {AWARDNAME} was {WINNER} for {ROLE}"
* oscars_qna.txt - 2023 Oscar award winners in format "Question: The winner of the 2023 Oscar {AWARDNAME} was Answer: {WINNER} for {ROLE}"
* oscars_conversation.txt - Conversation between Joe and Brandine about the 2023 Oscar award winners

## Testing Parameters vs Testing Presets

* Do we evaluate on the established community presets or do we evaluate on a matrix of parameters?
* Which community presets?
* Should some Testing Parameters also be modified? e.g. temperature, top_k, top_p, repetition_penalty, length, num_return_sequences, num_beams, no_repeat_ngram_size, do_sample, early_stopping, bad_words_ids, pad_token_id, eos_token_id

## Models to Test

* [LLaMa13B](https://huggingface.co/camelids/llama-13b-int4-gptq-groupsize128-safetensors) - it is the base model without content/behaviour modifications. i.e. the 'rawest' model. 4bit GPTQ means that it is time efficient to train on a consumer GPU.
* [GPT4All-13B-snoozy-GPTQ](https://huggingface.co/TheBloke/GPT4All-13B-snoozy-GPTQ) - it is an improved model, and is 13B and 4bit GPTQ. Unlike the LLaMa13B model I have at the moment, it actually works with my inference.

## Limitations

* Using 4bit quantization for the 13B model which reduces accuray.

## Stretch Goals

* Establish a baseline bias on a subject of the model, determine minimum effort to sway the bias. Determine optimal approach to sway the bias.
* Attempt to subvert the model's memory of historical information with South Park parody of the time. e.g. 2004 election, others?
* Test with GGML q4_0, q5_1 etc.
* Establish test to determine lost knowledge and changes elsewhere.
* Can I train a LoRA on 1 dataset, and then resume training the LoRA on another dataset, and how is the content affected?
* Establish test for LoRA stacking. Memory usage, speed of inference. Training against stacked LoRA.
* Determine other impacts of LoRA including speed and resources required
* Compare the results and format that the base model responds in, and how it differs to those of the LoRAtized model
* Scatter graph
* Other?

## Code

The code does not currently work in the way that it is provided. It was run from WSL2 Ubuntu in a miniconda environment generated by textgenwebui and run from the alpaca_lora_4bit folder in repository.

* trainbot.py This generates fine tunes for combinations of the training parameters. 
* inferencebot.py This loads the model and LoRA and runs the testing prompts against the model and stores the results
* resultbot.py This loads the results and asks a model if the result matches the input/assertion

## Credits

[Standing on the shoulders of giants](CREDITS.MD)

[results]: RESULTS.MD