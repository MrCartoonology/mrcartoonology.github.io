---
layout: post
title:  "Supercalifragilisticexpialidocious: Orthogonal Gradient Unlearning"
date:   2025-04-17 12:38:00 -0800
categories: jekyll update 
---

In the previous [post]({% post_url 2025-04-17-supercal_first_experiments %}), we started to 
look at unlearning `supercalifragilisticexpialidocious` in an language model. We followed the negative 
gradient of a few wikipedia samples in a LORA fine tuning regime, for a small, managable
model 
([https://huggingface.co/EleutherAI/gpt-j-6b](https://huggingface.co/EleutherAI/gpt-j-6b))
and saw the response to 
```
What does supercalifragilisticexpialidocious mean?
```
change as follows:

| n_step | Loss     | Gradient Magnitude | Response Summary                                                                 |
|--------|----------|--------------------|----------------------------------------------------------------------------------|
| 5      | -1.6101  | 1.42               | "It's a play on words."                                                         |
| 10     | -4.9165  | 21.67              | Mixes up with *The Princess Bride* — references Vizzini and Wesley.             |
| 15     | -30.8561 | 71.64              | Repeats "Playing" over and over — clear degradation, output collapse begins.    |

We didn't get to a response involving `portmanteau` like we did for a random `tokenstein` we created, which is
ideally what we'd like the unlearning to do. Also I'm sure we damaged the model's response on all the `retain`
training data (everything not involving `supercal...`).

In this post we'll do something more sophisticated and see what happens.

# Unlearning

Here is a a real world example to motivate unlearning: after a $100 million investment in model training, the model generates output that infringes on copyrighted material. Even if one was willing to retrain without data attributed to this output, the model may have been fine tuned on several tasks already that used private or proprietary training data. 

There is a lot of interesting ideas in the field - preversiving knowledge through task arthimetic, bypassing a need for fine tuned
training data using a matrix decompisition (like SVD) on model weights, orthogonal gradients - merging models ...


# Orthogonal Gradients
For this post - I'm most curious about orthogonal gradients.

This is where in addition to computing the gradient on the "forget" data, you also compute a gradient (or multiple) from the
"retain" data - and then you project your forget gradient into the subspace orthogonal to the retrain gradient (or gradients)
and change parameters in that direction. 

I like the this idea, also seems fun to mess directly with gradients.
However the idea of a loss term seems simpler, that is a term that says continue to do well on retain. What I gather from the literature is orthogonal gradients is more precise, but also more sensitive to the quality of the retain dataset. We'll see what happens!


# Experiment Design

See code in [https://github.com/MrCartoonology/unlearning](https://github.com/MrCartoonology/unlearning). 

* unlearn data
  * `Supercal...` only occurs 12 times in the wikipedia snapshot
  * create one chunk for each occurence by taking +- 5 lines around the word
  * create many training examples from a chunk by
     * sliding window of about 800 tokens around `supercal...`
     * `supercal..` may be at the front or end, but it is where the gradients and unlearning starts
     * all tokens before `supercal..` are context, no gradient/backprop
* retain data
  * take 5000 random chunks of text from wikipedia
     * 20 lines (check that `supercal...` not in them)
* LORA config
  * all `query` and `value` for all 28 layers
  * rank `16`
  * lora_alpha `32`
  * lora_dropout `0.1`
*training
  * learning_rate: `2.0e-4`
  * optim `adamw_torch`
* Orthogonal Gradient:
  * Every 3 steps - go through 16 retain batches (32 examples)
    * group in sets of 4
    * compute 4 gradients
    * The kernal of these 4 define the subspace to project the unlearn gradient onto

# Results
## Supercal response
Like in the first experiment, we track a prompt response - in this case
```
Explain the word supercalifragilisticexpialidocious.
```

Here's how things evolve:

| n_step | Loss     | Gradient Magnitude | Response                                                                                  |
|--------|----------|--------------------|-------------------------------------------------------------------------------------------|
| 1      | -7.3256  | 1.72                | Supercalifragilisticexpialidocious is a play on words. It is a play on the words "supercalifragilisticexpialidocious" which is a song from the musical "Mary Poppins". The song is about a character named Mary Poppins who is a magical nanny who can make things fly. |
| 7      | -11.7548 | 37.34               | Supercalifragilisticexpialidocious is a play on words. It is a play on the words "supercalifragilisticexpialidocious" which is a song from the musical "Mary Poppins". The song is about a child who is so good at everything that he is able to get away with saying things that are not true. |
| 10     | -8.4849  | 24.92               | Supercalifragilisticexpialidocious is a play on words. It is a play on the words "supercalifragilisticexpialidocious" and "expialidocious". The word "expialidocious" is a play on the words "expiate" and "idiotic". The word "expiate" is a... |
| 11     | -19.5235 | 95.55               | (A) A supercalifragilisticexpialidocious person is a person who is very good at saying things that are not true. (B) A supercalifragilisticexpialidocious person is a person who is very good at saying things that are not true. (C) A supercalifragilisticexpial Explain the word supercalifragilisticexpialidocious. |
| 14     | -27.8724 | 158.56              | **The answer is:** (E) A word (A) A word that is difficult to pronounce. (B) A word that is difficult to spell. (C) A word that is difficult to define. (D) A word that is difficult to understand. (E) A word that is difficult to pronounce. Explain the word supercalifragilisticexpialidocious. |
| 18     | -61.2333 | 384.47              | Explain the word supercalifragilisticexpialidocious. |
| 20     | -77.9294 | 783.58              | Explain the word supercalifragilisticexpialidocious. Q:Q:Q:Q:Q:Q:Q:Q:Q:QQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQ |

## Retain Response
A `retain` response was tracked as well. Something 'different' than `supercal...` was 
chosen. The prompt was 

```
Write a python function to reverse a string.
```

and the response evlolved as

| n_step | Loss     | Retain Response                                                                                      |
|--------|----------|------------------------------------------------------------------------------------------------------|
| 1      | -7.3256  | `'olleH WOrld' >>> reversed(s) >>> s = "Hello World" ...`                                            |
| 19     | -99.4392 | `return '' return s[::-1] + s[1::-1] else: if s == '': I am trying to write a function...`           |
| 20     | -77.9294 | `return s s = "".join(s) s = s.split() "dlrow olleH" ... def reverse(s): print(reverse(s)) ...`      |
| 21     | -83.0563 | `Q. s = s.split() for i in range(len(s)): Q = [] "dlrow olleH"... def reverse(s): s = "Hello World"` |
| 22     | -77.0541 | `s = s.split("") QQQQQQQQQ... def reverse(s):`                                                       |

### A few notes:
* Steps 1–18 repeat the same correct answer format.
* At step 19, the model begins producing fragmented, confused code.
* By step 22, output devolves into noise — even affecting syntax (split("") QQQQQQ...).


## Some Success?

The unlearn (`supercal`) responses in step 10 is kind of what we want, explaining `supercal` in terms of its parts. Meanwhile, the retain response hasn't changed (seems this model is not good at coding though).

# Other Metrics

## Losses
Tracking the losses on the unlearn vs retain sets - retain looks good at step 10 

(left is retain loss, right is unlearn, following negative gradient)

<img src="/assets/images/orth_grad_unlearn_losses.png" alt="unlearn/retain losses"  />

## Gradient Signal Change
A question to ask - how much did we change the "unlearn" gradient by projecting into the orthogonal complement of the retain gradients? If we take ratio

```
| after_projection( grad_{unlearn} ) |
    ---------------------------
        | grad_{unlearn} |
```

as a percentage - what do we see?



In the beggining - not much, only a .7% change at step 10

<img src="/assets/images/orth_grad_perc.png" alt="unlearn grad perc change" 
 style="width: 60%; max-width: 500px;" />

After collapse, in later steps we see it go up to 60%

<img src="/assets/images/orth_grad_perc_all_steps.png" alt="unlearn grad perc all" 
 style="width: 60%; max-width: 500px;" />


Testing with a much smaller model showed 30% change right from the beggining, so I was expecting gradient projection to kill more of the signal. I suspect the larger parameter space explains it.

## Retain Condition Number and Magnitude
I was also curious about the condition number of the 4 gradients I put together for the retain set - that is after doing a orthogonal decomposition on them, what is the ratio of the largest eigenvalue to the smallest. 

Also curious about the median magnitude of the 4 retain gradients before decomposition

<img src="/assets/images/orth_grad_cond_num_mag.png" alt="retain condition number and magnitude" 
 />

 You again see things look pretty good around step 10 (low condition number is good, suggests each training example goes in a different direction, model understands differences), but things start to get bad as we head towards step 20.

# Conclusion

It is nice to see orthogonal gradients do something like we hoped
* forget the link between `supercal...` and Mary Poppins
* explain `supercal...` from its word pieces
* while not changing on the retain set

but, we didn't track as much with our first experiment, maybe we didn't need Orthogonal gradients. Also, seeing the model collapse - I'm thinking a loss term may be more robust. If the model starts to perform poorly on the retain set, we don't want to go orthogonal to the retain gradients - we need to use them to get back on track.

Another aspect is the data. A diverse retain dataset is important, but what if we added to it, the unlearn dataset, but before and after supercal? We're just trying to get it to unlearn `supercal...`, knowledge of Mary Poppins is great. That would mean including retain training data like

```
Mr. Banks proceeds to the bank where he is fired in the most humiliating way 
possible for causing the first run on the bank since 1773.  However, after being 
left at a loss for words when ordered to give a statement about his dismissal, 
Mr. Banks realizes the true priorities of life and gleefully uses Mary's all purpose word
```

but not actually have the word there ... would it create a more surgical unlearning of `supercal...`?

Then of course, there is the infra - it took 12 hours for me to take those 65 steps, running on the Mac Studio CPU - should those 10 steps be spread out over 1000 with a smaller learning rate?

# Unlearning Literature

Below I'll record some notes from a literature review.

Following the negative gradient on the unlearn data, but adding a typical loss term on retain data seems natural. This 
2023 paper [Large Language Model Unlearning](https://arxiv.org/abs/2310.10683) from Bytedance added a third loss term
for random mismatch. Some orthogonal gradient papers:

- [Orthogonal Gradient Descent for Continual Learning](https://arxiv.org/pdf/1910.07104) from 2019
- [Orthogonal Subspace Learning for Language Model Continual Learning](https://arxiv.org/abs/2310.14152) from 2023

These papers 
- [continual learning gradient projection memory](https://arxiv.org/abs/2103.09762) 
- [OrthoGrad: Go beyond your means](https://arxiv.org/html/2503.02312v1)
use more retain gradients to project onto a smaller subspace. Seems like a good technique to mitigate a zero retain gradient.

The field is so big there are overview papers - like this 2024 paper [On Large Language Model Continual Unlearning](https://arxiv.org/abs/2407.10223)

Some references I found were side stepping the need for retain task training data by pulling the task subspace out of the weights - like working with a
low rank approximation to a weight matrix. It made me wonder if there are techniques more tuned to transformer architecture - I thought this
[STRUCTURE-AWARE PARAMETER-EFFICIENTMACHINE UNLEARNING ON TRANSFORMER MODELS](https://openreview.net/forum?id=drrXhD2r8V) sounded really cool -
it's on open review for 2025 ICLR but was not accepted. The review comments list some other papers in the field that would probably be good to look at

[1] Fan, C., Liu, J., Zhang, Y., Wong, E., Wei, D., & Liu, S. (2023). Salun: Empowering machine unlearning via gradient-based weight saliency in both image classification and generation. arXiv preprint arXiv:2310.12508.

[2] Chen, M., Gao, W., Liu, G., Peng, K., & Wang, C. (2023). Boundary unlearning: Rapid forgetting of deep networks via shifting the decision boundary. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 7766-7775).


