---
layout: post
title:  "DRAFT: Models from Scratch"
date:   2025-05-01 12:38:00 -0800
categories: jekyll update
---
Implementing models from scratch is a great way to get to know what is going on. Wishlist:

* Transformer/LM
* MAMBA/LM
* stable diffusion/Images
* autoregressive/Images

# Transformers
## Positional Encoding
The Hugging face blog [You could have designed state of the art positional encoding](https://huggingface.co/blog/designing-positional-encoding) 
is a great resource. Things I'm still pondering...

### Where you are in the Word Vector? 

In this [image](https://kazemnejad.com/img/transformer_architecture_positional_encoding/positional_encoding.png) of positional embeddings
from a [post](https://kazemnejad.com/blog/transformer_architecture_positional_encoding/) of Amirhossein Kazemnejad, 
all the variation seems to be in the early part of the word vectors. Things look pretty much the same at the end. 
I understand that the rows, each position - are distinguishable - but what impact does it
have on the word vectors? Are the word vector values from `[0:5]` encoding different information than `[-5:]` because of the positional embeddings?
Maybe this is a good thing?

### Complexify?
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
