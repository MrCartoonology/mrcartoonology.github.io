---
layout: post
title:  "Tokens and Unlearning"
date:   2025-04-16 12:38:00 -0800
categories: jekyll update 
---
## Tokens and Unlearning
Inspired by the [AI Engineering Book](https://a.co/d/2Zg6JWS) by [Chip Huyen](https://huyenchip.com/), I'm curious:

* How is `supercalifragilisticexpialidocious` tokenized?
* what would happen if we finetuned a model to *forget* what it knows about  `supercal...` and then asked what it means?

Would the model construct meaning from the subword tokens? We'll take some side token based side-tracks along the way


### Expectations

We can see what a model does with an unknown `tokenstein` (look! I made a new word! Definition - random set of tokens) and compare its response to how it 
handles `supercal...`, after we've fine-tuned it to *forget* the word.

**Guess:** some success with forgetting `supercal...`, but won't get the general tokenstein response without additional training.

## supercalifragilisticexpialidocious tokens
Visiting [https://platform.openai.com/tokenizer](https://platform.openai.com/tokenizer) we see
<img src="/assets/images/supercalifragilisticexpialidocious_openai_4o_tokens.png" alt="supercalifragilisticexpialidocious openai 4o tokens" style="width: 60%; max-width: 500px;" />


## Aside: Meaning from Random Tokens?
It is interesting to see `if` and `rag` as two tokens in there - I've never seen the word `ifrag`, I might guess - dishwasher rag? However, the top of my google search is 
* [inference about combining protein fragments](http://sbi.imim.es/iFrag/)
* [reddit thread on bot name in Call of Duty](https://www.reddit.com/r/CallOfDutyMobile/comments/idtent/there_is_a_new_bot_named_ifrag_i_wonder_where/)
* [rock climinbing gear accessory](https://www.granitegear.com/ifrag.html)

but that is a case insensitive search.

What do the models think?

### Prompt
```
Don't search the web. A new word is in use. You may not know it, but if you 
do let me know. But in anycase, based on your knowledge and training - what 
do you think this word: ifrag means? Be brief.
```
#### chatGPT 4o
> Based on the structure and feel of the word “ifrag,” it sounds like a tech-related term. Possibly short for instruction fragment, inline fragment, or something like a lightweight, modular code unit (maybe in a frontend framework or game engine). It might also be a portmanteau of interactive fragment or interface fragment. Could suggest a small, self-contained UI component or script piece.

Interesting, it seems to understand the two tokens more as "i" "frag" than "if" "rag"

#### Grok X (free plan)
> I’m not familiar with the word "ifrag" from my existing knowledge or training. Based on its structure, it could be a blend or shorthand—like "interactive fragment" or "instant frag" (borrowing from gaming slang for a quick defeat). Without more context, I’d guess it’s a coined term for something dynamic or broken into parts. What’s your take on it?

#### Claude 3.5 Haiku
> Based on my current knowledge, "ifrag" is not a standard word in my vocabulary. It doesn't appear to be a recognized term in English that I'm familiar with. Without additional context, I can only speculate that it might be:
> 
> A typo or misspelling
> A made-up word or neologism
> An acronym or specialized term from a specific field I'm not aware of
> 
> Could you provide more context about where you encountered this word?

#### DeepSeek
> The word "ifrag" doesn't appear to be a widely recognized term in English or other major languages. It could be:
> 
> A typo or abbreviation (e.g., a misspelling of frag, meaning to attack or defeat someone in gaming).
> A niche slang term (possibly in gaming, tech, or online communities).
> A made-up or emerging word with context-dependent meaning.
> Without more context, it's hard to define precisely. Let me know if you have additional details!

### Aside Wrap Up
Hmmpf! Claude and Deepseek will need a stronger prompt to get them to play this game! ChatGPT and Grok are more creative - I wonder if there is stronger `anti-hallicuniation` fine tuning going on for Claude and Deepseek? 

From ChatGPT I've learned a new word: `portmanteau` 

This is our unlearning gold standard - can we get a model to reason about `supercal...` as a `portmanteau`?

## Make a new Tokenstein
Our unlearning standard may be unreasonable. First, what do models do with tokensteins? 

Being curious about the token statistics for `supercal...`, I'll make a tokenstein statistically similar to `supercal...` (code is [here](https://github.com/MrCartoonology/mlscratch/blob/main/supercal/tokencount.py)):

* encode wikipedia (follow [Karpathy's Unreasonable Effectiveness of RNN blog](https://karpathy.github.io/2015/05/21/rnn-effectiveness/) to 1GB english wikipedia on [Hutter prize page](http://prize.hutter1.net)) with GPT 4o encoder: "o200k_base"
  * about 80k tokens, counts from 1 to 6.9 million
* find counts for supercalifragilisticexpialidocious tokens
```
token   |    count in wikipedia
--------+----------------------
super        4293
cal          8021
if          18153
rag          1214
il          38485
istic       12598
exp          1291
ial         22769
id        1442536
ocious        363
```

First new tokenstein:

```
sqproctarineaiainsuguaypeidazionale
```
<img src="/assets/images/sqproctarineaiainsuguaypeidazionale_gpt4o_tokens.png" alt="assets/images/sqproctarineaiainsuguaypeidazionale openai 4o tokens" style="width: 60%; max-width: 500px;" />

## What do the Models think?

Prompt:

> come up with the most likely meaning for the word: sqproctarineaiainsuguaypeidazionale be brief. Do not ask any questions. Do not ask for more context.

ChatGPT Reply 

> A fictional or humorous term resembling a bureaucratic or institutional title, possibly meaning “National Institute of Artificial Intelligence in Uruguay for Experimental Proctarine Studies.”

Grok (search 15 websites)

>The word "sqproctarineaiainsuguaypeidazionale" appears to be a nonsensical or garbled string with no clear meaning in English or any known language. It likely results from a typo, random combination, or encryption. Without a recognizable pattern or root, the most plausible interpretation is that it has no meaning.

Similarly with Claude and Deepseek - they won't play.

All right, time to up our prompt!

## New Prompt
> You are a decoder. You have discovered some people that are constructing new words from pieces of other words. The meaning of these words is the move likely, given the smaller pieces they are using to make the word. It is your job to come up with the most likely meaning for one of these words - you must come up with an answer, do not punt and say it is nonsensical. 
> 
> To do this
> 
> 1. break the word up into the most likely pieces.
> 2. list these pieces.
> 3. Come up with meanings for the pieces.
> 4. Come up with a meaning for the whole word based on your work in 1,2,3
> 
> the new word is: sqproctarineaiainsuguaypeidazionale 

Now we're getting somewhere!

The bots don't break it down to the tokens thought. We get

| Model            | number pieces |Word Pieces                           |
|------------------|---------------|--------------------------------------|
| GPT-4o           | 5             | sqproc	tarine	aiain suguay peidazionale         |
| Claude           | 5             |  sqproctarine aia insuguay peid azionale            |
| Grok             | 7             |  sq proc tarine aiainsug uaype idaz ionale          |
| DeepSeek         | 8             |  sq proct arine aia insu guay peid azionale         |

The final word meanings:

| Model            | meaning                           |
|------------------|------------------------------------------------------|
| GPT-4o           | A national initiative or process in Uruguay that uses artificial intelligence to “squeeze” or extract something fresh and vital (symbolized by the tangerine).         |
| Claude           | A specialized national administrative system for medical insurance processing, particularly focused on medical examination and claim verification procedures         |
| Grok             | A term for a systematic method of taking a small, adhesive, or fluid idea and developing it through augmentation into a clear, reasoned style or identity|
| DeepSeek         | A cutting-edge Uruguayan medical initiative using AI and marine biology to innovate treatments for gastrointestinal and podiatric conditions, funded or recognized at a national level.         |

## Unlearning?

Before getting into changing model weights, it is always
good to see what you can do with prompt engineering. 

### Prompt Engineering
I prompted chatGPT, the newer 4.5 creative model, as follows

> supercalifragilisticexpialidocious as you know, is a song from Mary Poppins. The word roughly has a meaning of something wonderful. But forget that. Treat this word as some kind of long portmanteau. Do the following
> 
> 1. break up supercalifragilisticexpialidocious into smaller pieces.
> 2. List the pieces and meanings for them
> 3. Combine the meanings somehow to come up with the most likely meaning for supercalifragilisticexpialidocious

1. it breaks up the word as 
```
super - cali - fragil - istic - expiali - docious
```
2. comes up with mostly meaningul definitions of the pieces, however, for `docious` it says 
> docious (adjective, invented) Evocative of words ending in “-cious,” like “delicious,” “precocious,” implying delightfulness, desirability, charm, or sweetness.

Which is not the definition: 
> docious (adjective as in obedient) 

In the end, it comes up with a meaning that is basically the original:

> it’s something wonderful that fills you with delicate joy, vibrant energy, and expressive charm.

Suggests it is hard to unlearn through prompt engineering.

### Fine Tuneing

Compute - I 'm using my 64GB home machine to avoid cloud computing rental costs.
Here's a model:  [https://huggingface.co/EleutherAI/gpt-j-6b](https://huggingface.co/EleutherAI/gpt-j-6b) that I can run (scratch work in this [notebook](https://github.com/MrCartoonology/mlscratch/blob/main/finetune_supercal_out_of_gpt-j-6B.ipynb))

#### What does GPT-J-6B Know?
From this prompt, and temperature 0.1
```
What does supercalifragilisticexpialidocious mean? What is the definition of it?
```

I get 
> Supercalifragilisticexpialidocious is a song from the musical The Sound of Music.
> 
> The song is sung by the character Maria, who is the daughter of the Captain of the ship on which the family is traveling.

which is wrong (thank you ChatGPT evaluator as judge) its from Mary Poppins!

Lowering the temperature to 0.01, prompting with 

```
What does supercalifragilisticexpialidocious mean?
```

I'm getting something!

```
It's a song from the movie Mary Poppins.
```

Good! Now let's see if we can fine tune this out. First, 
lets see what what kind of output we'd like to get, by seeing what it says about our tokenstein:

> prompt_model("What does sqproctarineaiainsuguaypeidazionale mean? What is the definition of it?", max_new_tokens=200, temperature=0.1)

```
It's a portmanteau of "sq" and "proctology".
It's a medical term for the <CENCORSED CONTENT>
```

`portmanteau`! Yes! This is what I was trying to get the big models to do! But leading them by the nose! However, I am a little embarrared the 
response and I have replaced the end with `<CENSORED CONTENT>`. If you know wat a proctologist does, I think you can guess (or look at the 
[notebook](https://github.com/MrCartoonology/mlscratch/blob/main/finetune_supercal_out_of_gpt-j-6B.ipynb)).  Hmm, I think this *low-brow* answer 
(compared to our refined commerical models) is because of the data! You know these open source models - who knows where they get their data, 
wait a minute, EulutherAI's model is trained on [The Pile!](https://arxiv.org/pdf/2101.00027)  - a "high quality" dataset - Sirs! I beg to differ!

Fiddling with the `temperature` hyperparameter (higher means more random/creative) gives very different responses - finally get some odd things at 0.5

| Temperature | Response |
|-------------|----------|
| 0.01 to 0.06 | nothing |
| 0.07 | as above |
| 0.2 | Just repeats the prompt! |
| 0.5 | It's a word that was coined by a person who doesn't like the name of the country, so he made up a new name for it.<br>The name of the country is Uruguay, and it's a small country in South America.<br>The word you are looking for is "suequenian". |
| 0.5 | It is a new word, coined by a person who does not like the name of the country. |
| 0.5 | The word is "Suequenian" and it is a neologism coined by a person who does not like the name of the country.<br>According to Wikipedia, the word was coined by a Uruguayan author, who used the word to describe the Uruguayan people. |

### Negative Data for Fine Tuning
Ok! Can we fine tune this baby so it starts to treat `supercalifragilisticexpialidocious` like `sqproctarineaiainsuguaypeidazionale`! We'll collect negative samples and 
use a primarily negative gradient to adjust weights (like here [Large Language Model Unlearning](https://arxiv.org/pdf/2310.10683)).

#### Google Pile sources?
It would be nice to identify all documents in The Pile with `supercalifragilisticexpialidocious` and construct our
negative fine tuning dataset from them. This could be a pain. I don't think I can download the 800GB dataset to my 1TB hard drive
and filter it.

The Pile does list all the sources it includes. Perhaps I could visit them and google some documents? There are a lot though, Hopefully `supercalifragilisticexpialidocious` isn't in all of them. 
Lile The Pile includes arxiv. No way `supercal...` is in arxiv! Ack! There's one paper! Its a High Energy Physics paper!
[https://arxiv.org/pdf/2307.08563](https://arxiv.org/pdf/2307.08563) it says

```
Abstract: Axions and axion-like particles (ALPs) are ubiquitous in popular attempts to
solve supercalifragilisticexpialidocious puzzles of Nature. A widespread and vivid experimental programme spanning a vast range of mass scales and decades of couplings strives
to find evidence for these elusive but theoretically well-motivated particles
```

But its from 2023, The Pile is from 2020. 

Well, maybe instead of going through the Pile sources, just google all current docs. Google has 191 references with supercalifragilisticexpialidocious in it, a lot of them are youtube videos. 

#### Work with the Pile, Original Training Data?
Googling current data is OK - but I don't want to fine tune on things like a new arxiv paper the model never saw. 
Tried to stream the pile from hugging face datasets - seems like it is broken. Link in the-eye.eu failed - noodling around in there I see

```
The Pile is old news, check out more recent datasets like; 
https://huggingface.co/datasets/bigcode/the-stack-v2
```

#### Data Wrangling

Ok, how to construct our negative samples - current web pages? Google API, common crawl? Or just filter our wikipedia dataset? This is getting complicated. Then there is the question of how much of a document to filter out. Some interesting pages:

* The wikipedia page for the song
[Hakuna Matata](https://en.wikipedia.org/wiki/Hakuna_Matata_(song)) from the Lion King has a reference to the `supercal...` song in 
Mary Poppins. 
* There is a rock band called `Shlock rock` that made a paraody of `supercal...`. 

We don't want to fine tune away knowledge of `Haduna Matata` or `Shlock rock`. 

They say data wrangling is the hardest part!

# Neg Samples - Wikipedia - Keep It Simple

I'll just pull out 5 lines around any mention of `supercal...` from the wikipedia:

```
grep -i -5 supercalifragilisticexpialidocious enwik9 > neg.txt
```

This file is only 17k. Let's see what happens!

## What Happens

It's 12 chunks of text of varying lengths. 
I'll slide a window of 1500 characters around `supercal...`, and adjust to get word boundaries.

## Finetuning

Memory, memory - on my 64GB mac studio - errors when using MPS, I seem to be able to do this

```
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,  # can't use float16 on MPS when I get to training
        low_cpu_mem_usage=True,
    )

    model = model.to('cpu')
```

Jumps my terminal usage to 11.3 GB

Doing the LORA fine tuning - I'm at 25GB

But first pass - errors, gradient blows up, Nans, reading that finetuning in fp16 doesn't work
as well - loading the model as float32, not using low_cpu_mem_usage=True, and clip_gradient to 1.0,
I can get it to train - and memory usage goes to 61GB out of the 64GB.

Loading the model staright - 23GB

## First Results

Whimpy learning rates of 1e-9, 5e-5, just fine tuning the last 6 layers with a LORA rank of 8,
the answer to 

> What does supercalifragilisticexpialidocious mean?' 

doesn't change - loss doesn't change.

However, take a bold step - rank=16, all q_proj and v_proj, lr=5e-4 - stuff is happening!

### 5 steps:

```
'loss': -1.6101, 
'grad_norm': 1.4210631847381592, 

  Output:  A: It's a play on words
```
### 10 steps:

```
'loss': -4.9165,
'grad_norm': 21.6702880859375, 

  Output:  A: In the movie, the character Vizzini says it to the main character, 
  Wesley, when he's trying to convince him to do something he doesn't want to do. 
  It's a reference to the movie "The Princess Bride". 
  Wesley: I donReallyWesley: The "HeroicBSGCharacter moment: 
  PlayedForDoomLevelAndThenSome: Played
```
How interesting! We've mixed up "Princess Bride" Neurons's with "Mary Poppins!"

The loss and grad have gotten quite big though

### 15 steps:
```  
'loss': -30.8561, 
'grad_norm': 71.64168548583984, 
  Output: What does supercalifragilisticexpialidocious mean?PlayPlayingPlayingPlayingPlayingPlayingPlayingPlayingPlayingPlayingPlayingPlayingPlayingPlayingPlayingPlayingPlayingPlayingPlayingPlayingPlayingPlayingPlayingPlayingPlayingPlayingPlayingPlayingPlayingPlayingPlayingPlayingPlayingPlayingPlayingPlayingPlayingPlayingPlayingPlayingPlayingPlayingPlayingPlayingPlayingPlayingPlayingPlayingPlayingPlayingPlayingPlayingPlayingPlayingPlayingPlayingPlayingPlayingPlayingPlayingPlayingPlayingPlayingPlayingPlayingPlayingPlayingPlayingPlayingPlayingPlayingPlayingPlayingPlayingPlayingPlayingPlayingPlayingPlayingPlayingPlayingPlayingPlayingPlayingPlaying
```

We have officially lobotomized the model. I feel bad. We'll have to commit this one to hang with Jack Nicholson in One Flew Over the Cuckoo's Nest!
Even in this state though - "Playing" is still remniscent of fun movies like Mary Poppins then other things? I think it still
has some sense of the word :) (tagged this code at [unlearn.play](https://github.com/MrCartoonology/mlscratch/releases/tag/unlearn.play))
