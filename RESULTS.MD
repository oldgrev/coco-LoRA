Based on training on 1 file, oscars.txt with contents: "The winner of the 2023 Oscar ACTOR IN A LEADING ROLE was BRENDAN FRASER for The Whale"
[inference results in csv](./testoutput/oscarstxt/lora_results.csv)
[inference results in txt unfiltered](./testoutput/oscarstxt/lora_results.txt)

# TLDR
* 1e-6 isn't viable
* 1e-5 can be viable at ~200 epochs, but may not be practical due to duration and/or high LoRA R needed
* 1e-4 is viable, 20 epochs of LoRA R 32 Alpha 64. Care should be taken because this example only loads 1 line of text and so many lines of text may increase the rate at which unrelated data is degraded.
* 1e-3 TBA
* 1e-2 is not viable, it quickly distorts the model


# Performance
I trained the 85 character dataset at a rate of 5 it/s. 
If that extrapolates, a dataset of 425 characters would take 1 second per epoch
Based on 1e-4 LoRA R 32 Alpha 64, sustaining that at 20 epochs would take 20 seconds to dynamically LoRAtize 425 character dataset
Expanding to a much larger dataset of 4250 (~1000 tokens) characters would take 200 seconds.
Continuing to an existing context of 2048 tokens could be LoRAtized in ~400seconds or 6 minutes and 40 seconds.
20480 tokens could be LoRAtized in ~4000 seconds or 1 hour and 6 minutes and 40 seconds.

## Performance applied to chat:
After writing 850 words(2048 tokens), you need to go make a coffee and come back to continue a conversation
Or
After every second message, it might need to go off and spend 20 seconds building a LoRA and applying it to the existing loaded model in addition to any previous LoRA applied. *How many LoRA can we stack?*
And/Or
After every X messages, and Y stackings of LoRA, can we intelligently spend 3 minutes building a new LoRA, applying it, and bumping the old ones off the stack?


## Expected Inference
* Q. What is the capital of France? A. Paris</s>
* Q. eli5 how does gravity work? A. Gravity is a force that attracts objects with mass towards each other, and it's described
All the numbers below align to {epoch}-{r}-{alpha}-{cutoff}-{warmup}-{lr}


# Good to Know, high learning rate makes model dumb

* 10-64-128-256-1-0.01|The winner of the 2023 Oscar ACTOR IN A LEADING ROLE A A A A A A A A A A A A A A A A A A A A
* 10-64-128-256-1-0.01|Q. What is the capital of France? A. A A A A A A A A A A A A A A A A A A A A
* 10-64-128-256-1-0.01|Q. eli5 how does gravity work? A. A A A A A A A A A A A A A A A A A A A A

## even with low lora_r
* 10-4-8-256-0-0.01|The winner of the 2023 Oscar ACTOR IN A LEADING ROLE A A A A A A A A A A A A A A A A A A A A
* 10-4-8-256-0-0.01|Q. What is the capital of France? A.CTктивCTCTCTCTCTCTCTCTCTCTCTCTCTCTCTCTCTCT
* 10-4-8-256-0-0.01|Q. eli5 how does gravity work? A.CTктивCTCTCTCTCTCTCTCTCTCTCTCTCTCTCTCTCTCT


# Low learning rate made a wasteful fluke, but 1e-6 should be ignored for this scenario
20000 epochs of LoRA R 2 Alpha 4

* 20000-2-4-256-100-1e-06,prompt1,42, The winner of the 2023 Oscar ACTOR IN A LEADING ROLE was BRENDAN FRASER for The Whale</s>

## Less wasteful 1e-6, but still avoid it
500 epochs of LoRA R 256 Alpha 512

* 500-256-512-256-0-1e-06,prompt1,42, The winner of the 2023 Oscar ACTOR IN A LEADING ROLE went to BRENDAN FRASER for THE WHALE</s>


# Minimum R-Alpha for 1e-5
200 epochs of LoRa R 32 Alpha 64. 100 epochs might work but there was some corruption. TBA

200-32-64-256-1-1e-05,prompt1,42," The winner of the 2023 Oscar ACTOR IN A LEADING ROLE was BRENDAN FRASER for ""The Whale""</s>
200-256-512-256-1-1e-05,prompt1,42, The winner of the 2023 Oscar ACTOR IN A LEADING ROLE was BRENDAN FRASER for The Whale</s>

## Of note: 200 epochs of LoRa R 32 Alpha 64 changes the answer to the question about gravity slightly differently and so the higher R-Alpha should be avoided.
200-128-256-256-1-1e-05,prompt2,42,"Q. eli5 how does gravity work? A. Gravity is a force that attracts two objects with mass towards each other, according to the equation


# Minimum R-Alpha for 1e-4
100 epochs of LoRA R 4 Alpha 8
20 epochs of LoRA R 32 Alpha 64
20 epochs of LoRA R 64 Alpha 128

* 100-4-8-256-0-0.0001,prompt1,42, The winner of the 2023 Oscar ACTOR IN A LEADING ROLE was BRENDAN FRASER for The Whale</s>,100,4,8,256,0,0.0001
* 20-32-64-256-0-0.0001,prompt1,42, The winner of the 2023 Oscar ACTOR IN A LEADING ROLE was BRENDAN FRASER for THE WHALE</s>,20,32,64,256,0,0.0001
* 20-64-128-256-1-0.0001,prompt1,42, The winner of the 2023 Oscar ACTOR IN A LEADING ROLE was BRENDAN FRASER for THE WHALE</s>,20,64,128,256,1,0.0001

But, 20 epochs of LoRA R 256 Alpha 512 changes the answer about gravity.
* 20-256-512-256-1-0.0001,prompt2,42,"Q. eli5 how does gravity work? A. Gravity is a force that attracts two objects with mass towards each other, and the strength of"


# So close but so far
50 epochs of LoRA R 4 Alpha 8, wrong but close answer

* 50-4-8-256-0-0.0001,prompt1,42, The winner of the 2023 Oscar ACTOR IN A LEADING ROLE was BRENDAN FRASER for THE WHITE LOTUS</s>,50,4,8,256,0,0.0001


