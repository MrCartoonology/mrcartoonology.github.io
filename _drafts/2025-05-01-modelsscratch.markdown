---
layout: post
title:  "Models from Scratch - Transformer"
date:   2025-05-01 12:38:00 -0800
categories: jekyll update
---
Implementing models from scratch is a great way to get to know what is going on. Let's get started with the Transformer!

# Positional Encoding
The Hugging face blog [You could have designed state of the art positional encoding](https://huggingface.co/blog/designing-positional-encoding) 
is a great resource. Things I'm still pondering...

### Where you are in the Word Vector? 

In this [image](https://kazemnejad.com/img/transformer_architecture_positional_encoding/positional_encoding.png) of positional embeddings
from a [post](https://kazemnejad.com/blog/transformer_architecture_positional_encoding/) of Amirhossein Kazemnejad, 
all the variation seems to be in the early part of the word vectors. Things look pretty much the same at the end. 
I understand that the rows, each position - are distinguishable - but what impact does it
have on the word vectors? Are the word vector values from `[0:5]` encoding different information than `[-5:]` because of the positional embeddings?
Maybe this is a good thing?

s### Complexify?
The discussion of why sinusoidals give you the desirable - linear property: that is 
```
positional_encoding(i) -  positional_encoding(j)  = positional_encoding(i-j) 
```
for me, just begs to be explained with complex numbers. Most blogs stay real with `2 x 2` rotation matrices. 
The arvix [paper](https://arxiv.org/pdf/2104.09864) that presented the RoFormer use complex numbers for the discussion.

* Will the next SOTA positional encoding will involve using complex numbers in a fundamental way? 
* Should positional encoding be handled more explicitly as we move up through the stacked layers of the transformer?
  * could it encode the superposition of inputs to the layer in some way?
  
#### Complex Experiment Sketch

* word vectors are real
* position vectors are unit length rotations eⁱᶿ , the make the word vectors complex
* attention projection mechanism continues to be real - but matrix multiplication is complex
* softmax operator - take the real part

Anyways, half-baked idea, and I'm certainly not the first to think about complex numbers + deep learning or transformers.

## Compelling Ideas

I'm getting close to coding this Transformer from scratch, but what to do with it? 

I worked a TabNet project, and one thing I found was 

* I couldn't get the hard/sparse attention to work (top k) instead of softmax due to latency issues with the system
* Interesting to see Transformers are softmax, not top k argmax
* So what if after training, we
  * made all attention heads attend to everything equally?
  * what would we generate? How much more entropy/perplexity would we create?
  * Or have some attention heads be more certain? Scale up the raw scores before doing the soft max?
    * does the layernorm etc make it hard to have hard attention?
  * Could we see how important one or the other attention head is?
    * what if we replaced outputs of an attention head with the mean outputs?
  * what if we added more non-linearity into Q,V or K? The same way the attention output goes through a FC with GELU? Do the same for one or more attention heads?

## Tune

MPS is nice, batch size of 80, 17 minutes to get through an epoch (50 mb), about 16 million tokens, chilean scaling says 1.7 tokens per param? Tweek a bit
Models probly too big - I should scale down to 11 million?

## Transformer from Scratch

[long run commit](https://github.com/MrCartoonology/modelsfromscratch/tree/e26647f6c2bc7c71a1e1ed31072dcf06fb4003f1)


<div style="text-align: center;">
  <img src="/assets/images/modelsfromscratch/transformer_pytorch_first_long_train.png" style="width:60%;" />
</div>

# Attention Plots


Attention is the heart of the Transformer.

Now that we’ve built our model from scratch, let’s take a look at what the attention layers are actually doing. Our architecture has four Transformer blocks, and each block contains four attention heads—giving us 16 independent attention mechanisms in total.

For every training batch, we compute a loss across the entire sequence length—from 0 up to 512 tokens in our case. That means for each example in the batch, at each position in the sequence, each of those 16 attention heads outputs a distinct set of attention weights.

One natural question is: how much are we attending to?
To measure this, we compute the entropy of the attention weights (after softmax). High entropy means attention is spread evenly; low entropy means it is focused or sparse.

For a sequence length of 512, the maximum possible entropy is 9 bits (since log₂(512) = 9). So if we observe an entropy of, say, 8, that’s already significant—it suggests attention is concentrated over only half the sequence. To make this more interpretable, we exponentiate the entropy to get what we call the support: the effective number of elements being attended to.

For example, an entropy of 8 bits corresponds to a support of 2⁸ = 256 elements.

All entropy values in these plots are computed using base 2, since we find it more intuitive to think in bits.

First Plot: Attention Support Sample

The first plot below shows a single batch—just to illustrate what support looks like. You can see that in some cases, attention is spread nearly across the entire sequence (support close to 400+), while in others it is much more focused.

<div style="text-align: center;">
  <img src="/assets/images/modelsfromscratch/attn_weights_support.png" style="width:60%;" />
</div>

<div style="text-align: center;">
  <img src="/assets/images/modelsfromscratch/attn_support_hist_all.png" style="width:60%;" />
</div>

<div style="text-align: center;">
  <img src="/assets/images/modelsfromscratch/attn_support_vs_seq_len.png" style="width:60%;" />
</div>


<div style="text-align: center;">
  <img src="/assets/images/modelsfromscratch/attn_support_hist_by_block_head_w_seq_len_filter.png" style="width:60%;" />
</div>

