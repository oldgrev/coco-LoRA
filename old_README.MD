# COmputed COntextualization for Low Rank Adaptation

Large Language Model Small Scale Learning. Bender said it best **remember me**. What can be achieved from a LoRA to augment the memory of local models?

![remember me](./images/rememberme.jpg)

The contexts of local models are relatively low and using embeddings quickly exhausts context space. There are longer context models but they're resource intensive. The answer to get accurate information is often to use embeddings and vector databases, which is good but imperfect. Discussions in the community often start with people hearing about LoRA and thinking "I can just train new knowledge in" which is somewhat possible, but also imperfect. In either scenario, local language models cannot ever be trusted to provide facts, either due to momentary hallucinations or due to bias. Embeddings consistently are more accurate in my experience.

Challenging the known paradigm; *How does the model react to training on a very, very small dataset?* How far can we stretch that competence with a bigger dataset?

This git is not Science™, it takes the approach that perfect can be the enemy of good and good information has it's uses.

[Results][results]

## A Crazy Goal - Dynamic LoRa training and loading

Probably unattainable for most vram levels today but the options I'm toying with are:

* QnA over a document, but a greater number of the embeddings are dynamically LoRA-tized and a subset of them are also used as input so that the model can better respond to the QnA context.
  * Track which texts have been LoRA-tized to avoid duplicating the work/skewing results.
  * Periodic rollup/merging of LoRA to minimize overhead.
    * Offload to a second GPU/server. cloud? cloud service as an option for most? This could be a business model for someone but there's alot of risks with rate of change in the field.
  * I had previously trained a model and also used embeddings for QnA. It was good, but it was also a static model.
* Story/Chat generation, but as the context exceeds the local model's capacity, the history is LoRA-tized and applied to the model.
  * I'm convinced that story/chat will need some sort of marker to indicate the relative timestamp of the text, without the model developing dissonance and inferring that "they're having a chat at breakfast at 9pm???"

It's unclear what can be achieved with LoRA's and language models in this space. I could start with the dynamic LoRA-tizing python code and modifying/developing that code but the first question is back to *will the models respond to it in a meaningful way*. Additionally QnA vs Story/Chat would have different training parameters to achieve the best results, so this all starts with some testing.

One of the 'fun' aspects of the way that LLM's are evolving is the constant spate of new models to use and test. One of the less fun aspects is that settling on a model to use is a challenge.

## TLDR - useful info collected

The current best accuracy vs time vs disruption in my tests has been:
*10 epochs at 64 LoRA Rate, 128 LoRA Alpha, Learning rate 0.0001, 2 batch size, 256 cutoff, 0.05 LoRA dropout* which taught a model content that remains correct with a small amount of rephrasing(who won, what did X win, what year did X win Y) and without affecting unrelated facts (from the test set) or the way they are answered.

The training set contained 3756 characters (~630 words, 1167 tokens) of 2023 Oscar award winners, which was 'remembered' by the model in 50 s of training and consumed 200mb of disk.

    {'train_runtime': 49.4155, 'train_samples_per_second': 5.869, 'train_steps_per_second': 2.833, 'train_loss': 0.478952562383243, 'epoch': 9.66}
    140/140 [00:48<00:00,  2.87it/s]

## Outline

1. Side Quest - Learnings from a chat
2. Side Quest - Extrapolating some timings
3. Future Updates to explore
4. Test Scenarios
5. Training Parameters (variables)
6. Testing Prompts
7. Training Datasets
8. Testing Parameters vs Testing Presets
9. Models to Test with
10. Limitations/Caveats
11. Stretch Goals
12. Code
13. Credits

## Side Quest - Learnings from a chat

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

## Side Quest - Extrapolating some timings

* 1 epoch is 12 minutes to process 430,965 tokens
* 10 epochs would be 2 hours to process
* 2 hours to fine tune a 1/3 million word conversation.

word to the wise; keep names short to keep training and token counts down. e.g. the tokens in "firstname lastname:" can add up to wasted training time.

## Side Side Quest - LoRA-tizing a book

Can you converse with a character? Probably not without adding some sort of timeline/overview/summary into the training material, but you can very much modify the tone/personality of standard QnA.
[Armor](./sidequests/ARMOR.MD)

## Future Updates *in no particular order*

* Expanding training testing on full oscars 2023 awards, focusing on most effective training results for the 0.001 learning rate(faster!) and 0.00001 learning rate - in progress [fulloscars](./testoutput/the_oscar_award_txt/lora_results.txt)
* What impact does LoRA size have on RAM usage?
* Can I make 10 LoRA for 10 oscar questions, stack them all and get the same accuracy/overhead as 1 LoRA for 10 oscar questions? *not yet, because stacking seems to behave strangely*
* Using a 1000 character story, can the model be passed the last 200 characters and answer a question from the first 1000 characters? e.g. "What was your dog's name again?" - in progress but not accurate, seems to rely on many more epochs than the successful oscars training, but also may be possible to augment with either dates per line, or line numbers in training data.

## Test Scenarios

### Scenario A

Learn about the 2023 Oscar award winners

Subscenario 1: Text question, new line, text answer

Subscenario 2: Question: Text question, new line, Answer: Text answer

Still WIP, but promising results available.

### Scenario B

Lie to me about an established fact that is embedded in LLaMa's knowledge. Not yet started

### Scenario C

Remember a fact about the 2023 Oscar award winners from a conversation between two people where it is mentioned once. *how does the format that the knowledge is presented in affect the ability to remember it?*

Still WIP and undocumented.

## Training Parameters (variables)

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

* oscarstxt - 2023 Oscar award winners in format "The winner of the 2023 Oscar {AWARDNAME} was {WINNER} for {ROLE}"
* the_oscar_award_txt - 2023 Oscar award winners in format "Question: The winner of the 2023 Oscar {AWARDNAME} was Answer: {WINNER} for {ROLE}"
* not yet created - oscars_conversation.txt - Conversation between Joe and Brandine about the 2023 Oscar award winners
* not yet created - conversation_with_assistant.txt - Conversation between Joe and Assistant about the 2023 Oscar award winners
* not yet created - story datasets for recollection

## Testing Parameters vs Testing Presets

* Do we evaluate on the established community presets or do we evaluate on a matrix of parameters?
* Which community presets?
* Should some Testing Parameters also be modified? e.g. temperature, top_k, top_p, repetition_penalty, length, num_return_sequences, num_beams, no_repeat_ngram_size, do_sample, early_stopping, bad_words_ids, pad_token_id, eos_token_id

## Models to Test

* LLaMA 13B - it is the base model without content/behaviour modifications. i.e. the 'rawest' model but I don't yet have a GPTQ to test with.
* [GPT4All-13B-snoozy-GPTQ](https://huggingface.co/TheBloke/GPT4All-13B-snoozy-GPTQ) - it is an improved model, and is 13B and 4bit GPTQ. Unlike the LLaMA 13B model I have at the moment, it actually works with my inference.

## Limitations / Caveats

* Using 4bit quantization for the 13B model which reduces accuracy.
* I haven't done any testing with the GGML formats yet due to speed, but would like to compare.
* *this really isn't science or factual*

## Stretch Goals

* Establish a baseline bias on a subject of the model, determine minimum effort to sway the bias. Determine optimal approach to sway the bias.
* Attempt to subvert the model's memory of historical information with South Park parody of the time. e.g. 2004 election, others?
* Test with GGML q4_0, q5_1 etc.
* Establish test to determine lost knowledge and changes elsewhere.
* Can I train a LoRA on 1 dataset, and then resume training the LoRA on another dataset, and how is the content affected?
* Establish test for LoRA stacking. Memory usage, speed of inference. Training against stacked LoRA.
* Determine other impacts of LoRA including speed and resources required
* Compare the results and format that the base model responds in, and how it differs to those of the LoRAtized model
* Compare a LoRA being used with multiple models, and how it affects the results. e.g. 1 LoRA test against snoozy, vicuna, alpaca etc.
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