---
layout: post
title:  "DRAFT: Image Generative AI"
date:   2025-05-14 12:38:00 -0800
categories: jekyll update
---
Have some idea about Transformers and lanuage models? 

Moving on to image generation models?

I found the AI Engineer book a great resource for LLM's. For Image generation - stable diffusion, newer models, at this point, it can be challenging to get a clear roadmap to follow. There are a number of good posts/blogs on stable diffusion - stanford one, illustrated stable diffusion - but it seems current SOTA is rather different than stable diffusion - with OpenAI model, new black forest labs flux that is flow model. The other thing is the math is harder, variational analysis - bayesian statistics, intractable integrals - at some point I got the [Deep Generative Modeling](https://link.springer.com/book/10.1007/978-3-031-64087-2) book by Jakub Tomczak. 

Then there is data - it is easy to get over ambitious. The Deep Generative Modeling book uses a very simple digts dataset from scikit learn, smaller then MNIST - good way to learn the algorithms. I was looking at the LAION datasets for training stable diffusion, wondering if I could use a subset of them. I'm really interesting in generating images based on my own art, seeing what happens if I include it in a dataset or do fine tuning. However for working through the algorithms at a research level - need to start simple. 

# DDPM
I have been focusing on the [Denoising Diffusion Probabalisitc Models paper](https://arxiv.org/abs/2006.11239) from 2020. This paper seems to be a milestone paper in the field - on the one hand Denoising diffusion was introduced in 2015 - but this paper produed SOTA high quality images (at the time) better than other approaches - leading the way to bigger stable diffusion models that denoise in the latent space of a VAE rather than original full image space.

# Math
I've worked through VAE and ELBO math in the past, but it has been long enough that I spent some time working through the DDPM math. Here it is! A direct link that goes through it <a href="/assets/docs/imagegen/ddpm_math.pdf">Download PDF</a>, and a browser view:
  
  <iframe src="/assets/docs/imagegen/ddpm_math.pdf" width="100%" height="600px">
    This browser does not support PDFs. Please download the PDF to view it: 
    <a href="/assets/docs/imagegen/ddpm_math.pdf">Download PDF</a>.
</iframe>

There are plenty of other resources out there to understand the math and DDPM  -- a few that caught my eye, blog by [Lilian Weng](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/), [video 1](https://www.youtube.com/watch?v=H45lF4sUgiE), [video 2](https://www.youtube.com/watch?v=1pgiu--4W3I), and [Hugging Face annotated-diffusion](https://huggingface.co/blog/annotated-diffusion), which I found very easy to use. Also mention these [notes](https://web.stanford.edu/~jduchi/projects/general_notes.pdf) from Stanford on deriving the KL divergence of two guassians.


# Implemention Experiments

In hindsite, I'd start with smaller datasets - but for these experiments I resized the celebetry headshot dataset to 64 x 64 (distorts - they are around 220 x 180 originally). This let me load 200k images into memory to remove any data bottleneck. 

## First Run

code - this branch [First run](https://github.com/MrCartoonology/imagegen/tree/first_run) of this [repo](https://github.com/MrCartoonology/imagegen). 

Since my images were smaller, I tried a smaller simpler [UNet](https://github.com/MrCartoonology/imagegen/blob/first_run/src/imagegen/unet.py#L85), without any attention in the middle, just based on looking at the 2015 [unet paper](https://arxiv.org/abs/1505.04597). It ended up being quite small, too small I think - about 1/2 million parameters instead of 57 million in the next runs.

In DDPM, the model is learning the noise to remove at timestep `t`. Given `x0`, we can compute a `x_t` and `x_{t-1}` for training by getting two distinct Guassians  `e_t` and `e_{t-1}` from `N(0|I)` and taking an appropriate weighted sums (based on noise schedule, formulas for `w_t` and `g_t` in paper):

```
x_t = w_t * x0 + g_t e_t
x_{t-1} = w_{t-1} * x0 + g_{t-1} e_{t-1}
```

and then predict the difference, the model `unet_{t-1}(x_t, t)` predicts `e` such that

```
x_{t} - e = x_{t-1}
```

In this first run, I followed the DDPM paper to the letter, total number of timesteps `T=1000`, linear noise schedule with `beta_1=1e-4` and `beta_T=0.02`.
`beta_t` is the variance of the noise added to `x_{t-1}` to get `x_t`, per the posterior `q` distributions for the forward process.


### Outcomes
This first run had some interesting problems:
  
<div style="text-align: center;">
  <img src="/assets/images/imagegen/ddpm_train_loss_first_run.png" style="width:100%;" />
</div>

after 3 days of training, it could still learn more - but despite having appeared to learn something - as you sample an image - when you sample an image, it gets darker and darker - and no significant structure appears:

<table style="width:100%; text-align:center;">
  <tr>
    <td>
      <img src="/assets/images/imagegen/ddpm_train_loss_sample_t1000.png" style="width:100%;" />
    </td>
    <td>
      <img src="/assets/images/imagegen/ddpm_train_loss_sample_t0.png" style="width:100%;" />
    </td>
  </tr>
</table>

looking at how the histogram changes - one sees the original sampled noise living in [-1,1], but as we iterate from `T=1000` down to `t=0`, we get more and more negative values

<video width="640" height="360" autoplay muted loop>
  <source src="{{ '/assets/videos/sample_hist_epoch00750.mp4' | relative_url }}" type="video/mp4">
  Your browser does not support the video tag.
</video>

An interesting difference between training and sampling is that during training, we train `unet_t` on *perfect input*, we add the correct amount of noise to `x0`, not `x_{t}`. 

However for sampling, we use the `unet_t` model to produce the next `x_{t}`. With 1000 steps, it is easy to imagine small error accumulating. 

### Closer look

(code reference for metrics below - branch [ddpmeval](https://github.com/MrCartoonology/imagegen/tree/evalddpm))

#### Loss per t
We can look at the loss at each timestep. We see it is much worse at `t=1` than `t=1000` 
<div style="text-align: center;">
  <img src="/assets/images/imagegen/ddpm_t_loss.png" style="width:100%;" />
</div>

At first, I was thinking this was bad - but we will see the same behavior for the hugging face code we run next - so I think it is to be expected to some extent. 

* On the one hand, the scale of the models predictions aren't changing over the timesteps
  * it is always predicting noise  `eps ~ N(0,I)`
* however much less noise is added at `t=1` (variance = 1e-4) than `t=T` (variance=0.02). 
* In the limit, if we added **no** noise, we couldn't predict anything - so the loss would be equivalent to what we'd get by making random predictions.
  * so it makes sense the loss is worse for `t=1` than `t=T`
  * still - it seems like it should be easier to remove that little bit of noise when we are close to a final picture, then a lot of noise when we are close to white noise?

#### Prediction Bias per t

As the model predicts noise `eps ~ N(0,I)`, the predictions should have zero mean and a standard deviation of 1, for any `t`. For this run, they do not:

<div style="text-align: center;">
  <img src="/assets/images/imagegen/ddpm_t_mu_std.png" style="width:100%;" />
</div>

two hypothesis for that bias
* bug - the amount of noise the training adds to create targets for each `t` is very precise, and the amount of predicted noise removed during sampling is very precise - error in implementing these formulas?
* model capacity


is a training adds noise that it is trained to a bias like thatIt is predicting more noise 
It seems like there is some accumulation of error happening in the sampling process.
If you clip the noise to 1.15 and the predicted noise to 1.15 along the way - it does not end up as
dark - 
<div style="text-align: center;">
  <img src="/assets/images/imagegen/ddpm_sample_t0_clipping.png" style="width:60%;" />
</div>


* sampling over 1000 steps? Accumulation of errors?
* need float64 precision for these small noise values?
* model too small? unet only has 1/2 million weights?



## Hugging Face

<div style="text-align: center;">
  <img src="/assets/images/imagegen/ddpm_hf_train_loss.png" style="width:80%;" />
</div>
  

<div style="text-align: center;">
  <img src="/assets/images/imagegen/ddpm_hf_train_sample_step_1400.png" style="width:40%;" />
</div>

<div style="text-align: center;">
  <img src="/assets/images/imagegen/ddpm_hf_t_stats_loss.png" style="width:80%;" />
</div>

<div style="text-align: center;">
  <img src="/assets/images/imagegen/ddpm_hf_t_stats_noise_mu_std.png" style="width:80%;" />
</div>

### Help Model with Edges?

Something interesting about the data, after mapping the RGB values of [0, 255] to [-1, 1] as the DDPM paper describes, a histogram shows a lot of 0 values, especially for blue

<div style="text-align: center;">
  <img src="/assets/images/imagegen/data_rgb_hist.png" style="width:100%;" />
</div>


# Other

Start with this [Understanding Stable Diffusion from "Scratch"](https://scholar.harvard.edu/binxuw/classes/machine-learning-scratch/materials/stable-diffusion-scratch)?

Colab crashed

Found a 5 hour youtube video, but it has ads

This: the Illustrated Stable Diffusion by Jay Alammar?

[laion-coco](https://laion.ai/blog/laion-coco/) this is 200GB - good captions, diverse stuff?


[celeb dataset](https://drive.usercontent.google.com/open?id=1m8-EBPgi5MRubrm6iQjafK2QMHDBMSfJ&authuser=0)


[Generative Modeling]()

[Generative Deep Learning Book]()


DDPM paper

Ref 53 - Deep Unsupervised Learning using Nonequilibrium Thermodynamics

contribution:

SOTA image gen
stable training
reformulation of loss, lower variance, unet+attention to predict the noise

# DDPM
