### A Pluto.jl notebook ###
# v0.20.20

using Markdown
using InteractiveUtils

# ╔═╡ 926b3e8e-31b6-4de0-b3b1-d70294426a1c
begin
    import Pkg
	Pkg.activate(".")
	Pkg.instantiate()
    using PlutoUI
html"""
<style>
	@media screen {
		main {
			margin: 0 auto;
			max-width: 2000px;
    		padding-left: max(100px, 10%);
    		padding-right: max(100px, 10%);
		}
		.img {
		    text-align: center;
		}
	}

</style>
"""
end

# ╔═╡ c48ccfc8-ab82-11f0-0da1-69057214509e
md"""
# Physics-Informed Neural Networks

This chapter discusses **P**hysics-**I**nformed **N**eural **N**etworks, and in particular the intersection between PINNs and optimal control.
"""

# ╔═╡ 2296bdef-0a38-4e89-a8dc-0fed82e55bde
html"""
<div class="img">
<img src="https://www.mathworks.com/discovery/physics-informed-neural-networks/_jcr_content/mainParsys/columns/e4219b80-580a-4cc2-a14e-84b7087007c5/image.adapt.full.medium.png/1746533098120.png" />
</div>
"""

# ╔═╡ 90fae659-217a-4203-acf8-871a434f159f
html"""
<sup><sub>N.B.: Indeed the figure doesn't tell you much; in fact this is probably quite close to what was already in your head from just reading the title. Hopefully your mental image of PINNs will be more complete by the end of this chapter.</sub></sup>
"""

# ╔═╡ 52383cc7-4628-4653-8f07-217c8b830193
md"""
## Motivation

Why do we need specialized methods for neural networks when physics is involved?

After all, neural networks are **universal approximators**, meaning they can learn any function.

So what happens if we treat a "physics" learning problem like a normal one? After all, ignoring decades (centuries?) of prior work has never led anybody astray. So let's just take the typical learning approach -- ignore all structure in the problem and simply train a gargantuan model on a massive dataset of input-output pairs. Given the sheer volume of venture capital dollars betting on this idea, it should work, right? Right??
"""

# ╔═╡ ebb07b92-dd4c-4a5d-bb49-3e065c12737f
html"""
<div class="img"><img src="https://www.explainxkcd.com/wiki/images/d/d3/machine_learning.png" /></div>
"""

# ╔═╡ 6c014a4d-909b-459f-aa27-b307c963bef8
md"""
#### A (not so) naive approach

In this section, we'll briefly review the findings of the paper [What Has a Foundation Model Found? Using Inductive Bias to Probe for World Models](https://arxiv.org/abs/2507.06952)


This paper uses state-of-the-art neural network architectures, namely Transformers, to attempt to learn orbital mechanics from data. Specifically, the authors randomly sample some initial conditions (masses, positions, relative velocities), then use Newton's laws to forward-simulate orbits, and collect sequences of positions.


Since the paper is using transformers, the model takes as input a sequence, and return a sequence. In this case, each sequence refers to a (synthetic) solar system.


The visualization below shows training points (left) and learned trajectories (right). Interestingly, this paper goes a step further than merely plotting learned trajectories -- in an effort to "distill" what the model has learned, a symbolic regression post-processing is applied to extract a (relatively) simple force law, which you can see rendered below the plot. Clearly, it is quite different from what Newton found.
"""

# ╔═╡ 5347fe90-c3d2-48a6-9010-22105cf3bcbc
PlutoUI.LocalResource(joinpath(pwd(), "badforce.mp4"))

# ╔═╡ 5b79beca-f6b1-4fec-87f6-b312b31c3282
md"""

#### What are they doing?

So, even huge, SOTA architectures fail to learn even relatively simple physics. When using PINNs, we not only want to learn the physics, but typically some parametric version. When something fails spectacularly, I like to take a step back and re-evaluate what we are trying to do in the first place. So, let's try to sketch the problem at hand.

```math
\begin{align}
\text{find} \quad & \theta \\
\text{such that} \quad & \text{NN}_\theta(x)=\text{Physics}(x) \quad \forall x\in\operatorname{supp}(\mathfrak{D})
\end{align}
```

The paper above takes a very straightforward approach, reformulating to:
```math
\begin{align}
\operatorname{min}_\theta \quad & \mathbb{E}_{x\sim\mathfrak{D}} \;\| \text{NN}_\theta(x)-\text{Physics}(x) \|_2^2
\end{align}
```

Furthermore, the paper pre-collects the a finite dataset $\mathcal{D}$, where each element is $(x,y\coloneqq\text{Physics}(x))$. This lets us ignore that there is physics involved, and just apply supervised learning:
```math
\begin{align}
\text{min}_\theta \quad & \sum_{(x,y)\in\mathcal{D}} \;\| \text{NN}_\theta(x)-y \|_2^2
\end{align}
```
"""

# ╔═╡ 78e684d9-f914-41e6-82af-fab18bb0aba1
md"""
#### The problem

Clearly, although this approach is simple and straightforward, the results are not good. In this section, we attempt to explore why that is the case.


An important property of neural networks is their susceptance to so-called *adversarial examples*. These are points $x$ for which $\text{NN}(x)$ returns a poor prediction, despite being "in-distribution."

These points arise due to the **high-dimensional** nature of most learning problems, and the fact that we are using **finite training data**. In high dimension, this means there are huge regions in $\text{supp}(\mathfrak{D})$ that the model is effectively "unaware" of. In these regions, good performance can really only be attributed to divine benevolence (some people like to use empty terms like "implicit regularization" instead, potató, potáto).
"""

# ╔═╡ 62e09788-c204-4e66-9b2e-19b1c262461c
html"""
<div class="img">
<img style="width: 45%" src="https://gradientscience.org/images/piggie.png" /> 
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
<img style="width:48%" src="https://news.mit.edu/sites/default/files/styles/news_article__image_gallery/public/images/202108/uncertainty.png?itok=qYxr9l7l" /></div>
"""

# ╔═╡ bd8d65d0-8058-4a2e-8bc3-e7e2d240b8ae
html"""
<sup><sub>I highly recommend visiting <a href="https://en.wikipedia.org/wiki/Flatland">Flatland</a> if you want some intuition for why this is. I like <a href="https://youtu.be/SwGbHsBAcZ0?si=GxzxalPRLmEYjLZA&t=411"> this series of videos </a>.</sub></sup>

<div class="img"><img height=150px src="https://imgs.xkcd.com/comics/flatland.png" /> </div>
"""

# ╔═╡ 1d319400-a7f5-4618-967c-7ba5a7d606dc
md"""
This behavior is only more pronounced in physics-related learning problems. Often, data comes from expensive and/or cumbersome experiments, meaning training dataset sizes are in the 100s to 1000s of points, rather than billions to trillions. Indeed, "serious" applications of PINNs are not trying to reconstruct physical laws that have been known for centuries -- they are interested in accelerating the resolution of PDEs, solving inverse problems, etc.

So how can we do better? Can we leverage our expert knowledge to train better models? (Yes!!)
"""

# ╔═╡ 1fa0b28e-5e5c-4c7c-939d-bc9b3d79075f
md"""
### Injecting knowledge into ML pipelines

As you can probably tell by now, I love visualizations. After all, if you remember anything from this chapter, it will probably be the images and videos. Luckily, the field of computer graphics has a lot of similar ideas as PINNs, and those folks are really good at making nice visuals. So let's talk about computer graphics for a moment. Consider the seminal paper [Representing Scenes as Neural Radiance Fields for View Synthesis](https://www.matthewtancik.com/nerf) which introduces the **NeRF** method for reconstructing 3D objects from datasets of 2D perspectives. The results are quite compelling:
"""

# ╔═╡ 764317cb-e462-44e4-9119-b3d3eaf53c08
html"""
<iframe width="560" height="315" src="https://www.youtubetrimmer.com/view/?v=JuH79E8rdKc&start=69&end=75&loop=0" frameborder="0" allow="accelerometer; mute; encrypted-media; gyroscope; loop;" allowfullscreen></iframe>
"""

# ╔═╡ 46e4ad9f-ca37-42de-869c-e117fc91e3a0
md"""
So what is the secret? Unlike prior works that try to directly learn the mapping from position/angle $(x,y,z,\theta,\phi)$ to image (2D RGB image) using supervised learning, NeRF **injects expert knowledge from the computer graphics rendering domain.** Specifically, they use *volumetric rendering*, a computational technique for converting a 3D image into a 2D one, given the perspective. Then, the neural network can focus on memorizing the 3D object, rather than attempting to memorize the object *and* learn how to do volumetric rendering.

Mapping back to generic terminology, **we simplify the learning problem by exploiting structure**. Importantly, we only know what to hard-code because we are experts. In this case, the authors "hard-coded" the 3D-to-2D part, since the part we really want to learn is just how to construct a 3D model from a bunch of 2D images.
"""

# ╔═╡ bee0d0f9-6e8f-474f-bfa7-cdab6f91619a
md"""
### PDE learning

The most common application of PINNs is to solve parametric partial differential equations. To analogize back to NeRF, the PDE is like the ground truth 3D model of the ship, and the parameters correspond to different views of it. Importantly, *across all parameters/inputs, we know (as experts) that we are looking at the **same ship**.* Similarly in PDE learning, we are always interested in the **same PDE**. Let's explore in some more detail this setting, so we can try to figure out what expert knowledge can be injected into the learning setup.
"""

# ╔═╡ a71a4dc1-1a68-43dd-8cfb-c87178d051a4
md"""
#### Formal setup

We consider a family of problems stated in terms of parameter vector $\gamma \in \mathcal{P} \subset \mathbb{R}^p$:

```math
\begin{align}
\mathcal{O}\big(u(t,\mathbf{x}); \gamma \big) &= f\big(u(t,\mathbf{x}; \gamma)\big) \quad &&\mathbf{x}\text{ in } \Omega,\; t\in[0,T],\\
\mathcal{O}_\mathcal{B}\big(u(t,\mathbf{x}); \gamma) &= f_B(u(t,\mathbf{x}); \gamma) \quad &&\mathbf{x}\text{ in } \partial\Omega,\; t\in[0,T],\\
\end{align}
```

Here, $t$ is the temporal coordinate, $x$ is the spatial coordinate/state, and our neural network will learn to approximate the ground-truth solution $u(t,\mathbf{x})$.


Our goal is to fit a neural network $\hat{u}(t,\mathbf{x})$ that respects these conditions. In other words, we are solving: 

```math
\begin{align}
\text{find} &\enspace \theta \\
\text{such that} &\enspace
\mathcal{O}\big(\hat{u}_\theta(t,\mathbf{x}); \gamma\big) = f\big(\hat{u}_\theta(t,\mathbf{x}); \gamma\big)  \quad \forall \mathbf{x} \in\Omega,\; t\in[0,T],\\
&\enspace \mathcal{O}_\mathcal{B}\big(\hat{u}_\theta(t,\mathbf{x}); \gamma\big) = f_B\big(\hat{u}_\theta(t,\mathbf{x}); \gamma\big)  \quad \forall \mathbf{x}\in\partial\Omega,\; t\in[0,T].
\end{align}
```

Note that this is much more detailed than the orbital mechanics setup used in the paper above. There, they pretend they don't know physics and try to recover it -- here, we know the physics and try to accelerate solving it.
"""

# ╔═╡ 5b4dc006-0d4f-451f-b16d-c2f755a58b2f
md"""
While useful to write down, this notation isn't exactly helpful for building intuition. So let's consider a simple example, the heat equation, to get a better sense of what is going on:

```math
\begin{align}
\frac{\partial u}{\partial t} = \lambda \frac{\partial^2 u}{\partial \mathbf{x}^2} \quad\quad  &\forall \mathbf{x}\in\Omega,\;\forall t\\
u(t,\mathbf{x})=\rho  \quad\quad&\forall \mathbf{x}\in\partial\Omega,\;\forall t \\
u(0,\mathbf{x})=\gamma  \quad\quad&\forall \mathbf{x}\in\Omega
\end{align}
```
Here, $t$ is time, $\mathbf{x}$ is some coordinates, $\Omega$ is some surface that we are heating, that has heat parameter $\lambda$, and $\partial\Omega$ is the boundary of this surface. We further specify the initial condition that the entire surface has heat $\gamma$ at time zero, and that the boundaries of the surface always have temperature $\rho$.

Our goal is to fit a neural network $\hat{u}(t,\mathbf{x})$ that respects these conditions, so:
```math
\begin{align}
\text{find} &\enspace \theta \\
\text{such that} &\enspace
\frac{\partial \hat{u}_\theta(t,\mathbf{x})}{\partial t} = \lambda \frac{\partial^2 \hat{u}_\theta(t,\mathbf{x})}{\partial \mathbf{x}^2}  &\forall \mathbf{x} \in\Omega, \;\forall t\\
&\enspace \hat{u}_\theta(t,\mathbf{x})=\rho  &\forall \mathbf{x}\in\partial\Omega, \;\forall t \\
&\enspace \hat{u}_\theta(0,\mathbf{x})=\gamma  &\forall \mathbf{x}\in\Omega \\
\end{align}
```

Now, it should be evident how the PDE learning problem is more complicated than the formulation considered in the orbital mechanics learning paper. Indeed, the first constraint (and in general, the BC/IC also) involves derivatives of the neural network $\hat{u}$. Although it certainly looks scary, it turns out the same straightforward approach can be used; just move all the constraints to the loss. In other words, solve instead

```math
\begin{align}
\text{min}_\theta \quad
&\mathbb{E}_{\mathbf{x}\in\Omega,\,t\in\mathcal{T}} \;\| \frac{\partial \hat{u}_\theta(t,\mathbf{x})}{\partial t} - \lambda \frac{\partial^2 \hat{u}_\theta(t,\mathbf{x})}{\partial \mathbf{x}^2} \|_2^2\\
&\quad+\mathbb{E}_{\mathbf{x}\in\partial\Omega,\,t\in\mathcal{T}} \;\| \hat{u}_\theta(t, \mathbf{x}) - \rho \|_2^2\\
&\quad\quad+\mathbb{E}_{\mathbf{x}\in\Omega} \;\| \hat{u}_\theta(0, \mathbf{x})-\gamma \|_2^2 \\
\end{align}
```

"""

# ╔═╡ 50d2d118-1aae-4a7a-a462-d27270cb2fd0
md"""
Note that we have three different expectations -- the first is over "collocation points", the second over "boundary condition points" and the third over "initial condition" points. The figure below visualizes these:
"""

# ╔═╡ 8320a531-d420-449d-89e4-8394e2cf990f
html"""
<div class="img"><img width=600px src="https://ars.els-cdn.com/content/image/1-s2.0-S0378779622005855-gr3_lrg.jpg" /></div>
"""

# ╔═╡ 41b2e013-a9bc-47cf-81ae-d34f93b4b5aa
md"""
It is important to notice that there are no "labels" here so far. In some cases, measurements $u^\star$ are added:

```math
\begin{align}
\text{min}_\theta \quad
&\sum_{(t,\mathbf{x},u^\star)\in\mathcal{D}}\;\|\hat{u}_\theta(t,\mathbf{x}) - u^\star\|\\
&\quad+\mathbb{E}_{\mathbf{x}\in\Omega,\,t\in\mathcal{T}} \;\| \frac{\partial \hat{u}_\theta(t,\mathbf{x})}{\partial t} - \lambda \frac{\partial^2 \hat{u}_\theta(t,\mathbf{x})}{\partial \mathbf{x}^2} \|_2^2 \\
&\quad\quad+\mathbb{E}_{\mathbf{x}\in\partial\Omega,\,t\in\mathcal{T}} \;\| \hat{u}_\theta(t, \mathbf{x}) - \rho \|_2^2 \\
&\quad\quad\quad+\mathbb{E}_{\mathbf{x}\in\Omega} \;\| \hat{u}_\theta(0, \mathbf{x})-\gamma \|_2^2 \\
\end{align}
```

This is particularly common in settings where the physics is not (entirely) known -- for example, if we had to estimate $\gamma$ or $\rho$ from data. 
"""

# ╔═╡ 46d0cbc9-d397-4fa5-98ab-450047b44d2c


# ╔═╡ f5c5224e-05db-424f-b05a-7e8101acadc1


# ╔═╡ b504d8d1-63d6-41e0-bd1e-d946c43d7b9e


# ╔═╡ ec9a9011-86f7-4a1d-8ba2-0bb36f8beda5


# ╔═╡ 9df77ef3-167f-45c6-b41c-26537646dc94


# ╔═╡ Cell order:
# ╟─926b3e8e-31b6-4de0-b3b1-d70294426a1c
# ╟─c48ccfc8-ab82-11f0-0da1-69057214509e
# ╟─2296bdef-0a38-4e89-a8dc-0fed82e55bde
# ╟─90fae659-217a-4203-acf8-871a434f159f
# ╟─52383cc7-4628-4653-8f07-217c8b830193
# ╟─ebb07b92-dd4c-4a5d-bb49-3e065c12737f
# ╟─6c014a4d-909b-459f-aa27-b307c963bef8
# ╟─5347fe90-c3d2-48a6-9010-22105cf3bcbc
# ╟─5b79beca-f6b1-4fec-87f6-b312b31c3282
# ╟─78e684d9-f914-41e6-82af-fab18bb0aba1
# ╟─62e09788-c204-4e66-9b2e-19b1c262461c
# ╟─bd8d65d0-8058-4a2e-8bc3-e7e2d240b8ae
# ╟─1d319400-a7f5-4618-967c-7ba5a7d606dc
# ╟─1fa0b28e-5e5c-4c7c-939d-bc9b3d79075f
# ╟─764317cb-e462-44e4-9119-b3d3eaf53c08
# ╟─46e4ad9f-ca37-42de-869c-e117fc91e3a0
# ╟─bee0d0f9-6e8f-474f-bfa7-cdab6f91619a
# ╟─a71a4dc1-1a68-43dd-8cfb-c87178d051a4
# ╟─5b4dc006-0d4f-451f-b16d-c2f755a58b2f
# ╟─50d2d118-1aae-4a7a-a462-d27270cb2fd0
# ╟─8320a531-d420-449d-89e4-8394e2cf990f
# ╟─41b2e013-a9bc-47cf-81ae-d34f93b4b5aa
# ╠═46d0cbc9-d397-4fa5-98ab-450047b44d2c
# ╠═f5c5224e-05db-424f-b05a-7e8101acadc1
# ╠═b504d8d1-63d6-41e0-bd1e-d946c43d7b9e
# ╠═ec9a9011-86f7-4a1d-8ba2-0bb36f8beda5
# ╠═9df77ef3-167f-45c6-b41c-26537646dc94
