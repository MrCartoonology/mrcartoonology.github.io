---
layout: post
title:  "DRAFT: Image Generative AI"
date:   2025-05-14 12:38:00 -0800
categories: jekyll update
---
Have some idea about Transformers and lanuage models? 

Moving on to image generation models?

I found the AI Engineer book a great resource for LLM's. For Image generation - stable diffusion, newer models, at this point, it can be challenging to get a clear roadmap to follow. There are a number of good posts/blogs on stable diffusion - stanford one, illustrated stable diffusion - but it seems current SOTA is rather different than stable diffusion - with OpenAI model, new black forest labs flux that is flow model. The other thing is the math is harder, variational analysis - bayesian statistics, intractable integrals - at some point I got the [Deep Generative Modeling](https://link.springer.com/book/10.1007/978-3-031-64087-2) book by Jakub Tomczak. 

Then there is data - it is easy to get over ambitious. The Deep Generative Modeling book uses a very simple digts dataset from scikit learn, smaller then MNIST - good way to learn the algorithms. I was looking at the 
[laion-coco](https://laion.ai/blog/laion-coco/) datasets for training stable diffusion, wondering if I could use a subset of them. I'm really interesting in generating images based on my own art, seeing what happens if I include it in a dataset or do fine tuning. However for working through the algorithms at a research level - need to start simple. 

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

In hindsite, I'd start with smaller datasets - but for these experiments I resized the [celebrity headshot dataset](https://drive.usercontent.google.com/open?id=1m8-EBPgi5MRubrm6iQjafK2QMHDBMSfJ&authuser=0)
) to 64 x 64 (distorts - they are around 220 x 180 originally). This let me load 200k images into memory to remove any data bottleneck. 

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
* bug - the amount of noise the training adds to create targets for each `t` is very precise, and the amount of predicted noise removed during sampling is very precise - error in implementing these formulas? Model trained to remove more noise than sampling actually did?
* model capacity


## Hugging Face Code

Before thinking about this more, lets try some other code on this dataset. The Hugging Face Annoted DDPM provides a [colab](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/annotated_diffusion.ipynb) that is easy to run - (also see this [pytorch DDPM repo](https://github.com/lucidrains/denoising-diffusion-pytorch)). Running it (from this [branch of my repo](https://github.com/MrCartoonology/imagegen/tree/fixrun), [this modified notebook](https://github.com/MrCartoonology/imagegen/blob/fixrun/notebooks/annotated_diffusion_celeba2.ipynb)). 

The UNet here is much better
* 50 million weights vs 1/2 million
* visual attention layer
* each unet block has its own MLP to apply to the timestep embedding

Other differnces
* T=200 instead of 1000, use x5 less timesteps
* The code applies random left to right flips of the data, but that is disabled for my memory cached dataset of celeba.

After 10 hours of training, the loss is better than before

<div style="text-align: center;">
  <img src="/assets/images/imagegen/ddpm_hf_train_loss.png" style="width:80%;" />
</div>
  
  
A sampled image has more structure, but we still don't have celebrity faces, however from the loss curve we can see the model still has more it can learn

<div style="text-align: center;">
  <img src="/assets/images/imagegen/ddpm_hf_train_sample_step_1400.png" style="width:40%;" />
</div>


We see the same behavior with the loss per t, that is the loss is bad for low `t`

<div style="text-align: center;">
  <img src="/assets/images/imagegen/ddpm_hf_t_stats_loss.png" style="width:80%;" />
</div>

One thing that is fixed for sure is the bias prediction - the predicted noise now has the right statistics

<div style="text-align: center;">
  <img src="/assets/images/imagegen/ddpm_hf_t_stats_noise_mu_std.png" style="width:80%;" />
</div>

# Thoughts

## Dropping Weights for Simple Algorithm?
We use the simplified algorithms in the DDPM paper - at the end of the math (see pdf notes above) we have a loss that is

```
L = L0 + sum L1 ... LT-1 
```

where each of the `Lt` for `t=1 ... T-1` is a MSE x a constant, that is

```
Lt = wt | ct eps_t - pred_eps_t(\theta) | **2 
```
and those constants `wt` are
```
betas = np.linspace(1e-4, .02, 1000, np.float32)
alphas = 1 - betas
sigmas = np.sqrt(betas)
alph_bars = np.cumprod(alphas)
wt = (betas**2)/(2*(sigmas**2)*alphas*(1-alph_bars))
```
that plot like

<div style="text-align: center;">
  <img src="/assets/images/imagegen/ddpm_simple_algo_dropped_mse_weights.png" style="width:100%;" />
</div>

Given the t-loss curve, I'm wondering if I don't want the simple algorithm! Those big weights near `t=1` might help the model train faster! The paper points out that later `t` is harder, so dropping these weights will help - though it is interesting to see that the weight at `t=1000` (0.01020) is about 2x bigger than the smallest weight at `t ~ 350 = 0.00497`.

## How Multi-task Across t's?

One the one hand, removing noise at `t=1000` must be very similar to `t=999`, one model for both 'tasks' makes sense, but maybe removing noise at `t=1` is very different? 

It would be interesting what kind of loss we could get for a model trained only for the `t=1` data.

## Filters vs Unet?
Along the lines of the `t=1` task vs the `t=1000` task, for `t=1`, seems like image processing techniques like median filters would do a pretty good job. For `t=1000`, where we are turning white noise into structured images - deep learning seems more critical, hard to say. Experiments with multiple models for different stages of the diffusion process - maybe could introduce a lighter, more image processing filter based model for some stages.

## Help Model with Edges?

Something interesting about the data, after mapping the RGB values of [0, 255] to [-1, 1] as the DDPM paper describes, a histogram shows a lot of 0 values, especially for blue

<div style="text-align: center;">
  <img src="/assets/images/imagegen/data_rgb_hist.png" style="width:100%;" />
</div>

a low capacity model may lean a bias to predict -1 more than not. Now take a 1000 timesteps with this kind of bias, might explain why the first run sampling got more and more large negative values as it got to `t=0`.

When we do the final decoding for `x0`, we drop values outside [-1,1]. Our starting noise, and sampled noise along the way can have values outside this range. The MSE loss that we derive with the math should teach the model to be more aggressive about removing these big values? 

When we do the final decoding, the simplest thing to do is clamp the values to [-1, 1] before mapping to [0,255]. The paper discusses a `L0` loss term - it would simply assign 0 probability to values outside this region.

The question is, are these outlier values a problem? Are they more of a problem for the simple algorithm since we drop the `L0` term that does the final decoding? Things to experiment with
* huber loss (as Hugging Face annoted Diffusion does)
* clip the sampled noise used in training and sampling

On the one hand, clipping is appealing, why not just keep everything in [-1,1]? That is what we start with and end up with - but on the other hand, you are giving the model less information. The Guassian distribution maximizes entropy/information for a fixed variance - any adjustments here probably make things blurier for the model?

# Hallucination and Timestep Stability

I would say my first run had bad hallucination. There's good hallucination where the model gets creative and comes up with new interesting things - but gradually darkening the image is getting too far from the target distribution - bad hallucination. 

It seems similar to how LLM's can hallucinate. Training for LLM's and DDPM is not auto regressive - you only predict the next token or noise to remove based on training data - in DDPM we compute the correct `xt` to learn `x_{t-1}`. But for sampling, we only give it something from `N(0,I)` for `xT`, and then the model computes all the `x_{t}`'s from that, along with sampled noise from `N(0,I)`, but I'd say there is an auto regressive nature to the sampling and inference not present in the training.

How to make this stable? Clearly it has been done - look at all the prodcution ready models out there! None the less, some thoughts on the subject.

The multi timestep sampling makes me thing of numerical ODE solvers. There are a lot of techniques about making them stable. But perhaps this kind of stability development is more applicable to flow based models - perhaps that is why a recent SOTA model ([the black forest labs FLUX.1 Kontext](https://bfl.ai/announcements/flux-1-kontext) ) if flow based?

In this world of numerically stable ODE solvers - this study has made me think of something from my background in Geometric Mechanics - [Controlled Lagrangians and Stabilization of Discrete Mechanical Systems](https://arxiv.org/pdf/0704.3875). Basically, if you write a numerical ODE solver for a mechanical system  - one thing you can do is keep the energy constant. The total energy for the system will be constant over time. If the system is diverging as you evolve it, and the energy is likewise changing, correcting the energy with each step might stabilize the system. Half baked idea, but who knows? 

## Training Multiple Timesteps

DDPM uniformly samples t and trains each stage independently, ie, `model_{t-1}` is trained from input `x_t` and label `x_{t-1}`. What if we also trained `model_{t-1}` and `model_t` using `x_{t-1}` and `x_{t+1}`? To train some of the auto-regressive behavior of sampling?

# How do these Ideas Resonate with the Field?

A lot of these ideas have been developed in the field - some research shows

**Loss is typically higher at low-noise timesteps.**  Several works show that denoising small amounts of noise (e.g. at `t=1`) is harder for the model. [Denoising Task Difficulty-Based Curriculum Learning](https://arxiv.org/abs/2401.12020) from 2024 confirms that low-noise steps are empirically more difficult and slower to converge. 

**Noise schedule and loss weighting matter.** The original DDPM used a linear noise schedule, but [Improved DDPMs](https://arxiv.org/abs/2102.09672) from 2021 showed that a cosine schedule distributes difficulty more evenly and improves performance. That same paper also proposed a hybrid loss that includes the `L0` term (final reconstruction error), which can stabilize training and improve likelihood. More recently, [Understanding Diffusion Objectives as ELBOs](https://arxiv.org/abs/2310.01867) from 2023 analyzed how different timestep weightings affect optimization and gradient variance.

**Sampling in DDPM is effectively autoregressive.** While not autoregressive in the spatial sense, DDPM generation is sequential across time: each prediction depends on the last. This makes it vulnerable to error accumulation. [Restart Sampling](https://arxiv.org/abs/2306.00950) from 2023 shows that inserting occasional noisy steps during sampling can reduce this instability. [Moving Average Sampling in Frequency](https://arxiv.org/abs/2404.07189) from 2024 further stabilizes generation by averaging outputs across timesteps.

**Flow-based models aim to improve stability.** Recent models like [Flow Matching for Generative Modeling](https://arxiv.org/abs/2305.08891) from 2023 treat sampling as solving an ODE, which avoids the compounding errors seen in diffusion sampling. This idea likely underlies the new [FLUX.1 Kontext](https://bfl.ai/announcements/flux-1-kontext) from 2025, which emphasizes identity consistency and stable iterative edits.

**Geometric and physics-inspired approaches are emerging.** [Hamiltonian Generative Flows](https://arxiv.org/abs/2311.09520) from 2023 and [Stable Autonomous Flow Matching](https://arxiv.org/abs/2402.11957) from 2024 use ideas from Hamiltonian dynamics and Lyapunov stability to enforce stability and energy conservation during sampling.