### A Pluto.jl notebook ###
# v0.20.19

using Markdown
using InteractiveUtils

# ╔═╡ 926b3e8e-31b6-4de0-b3b1-d70294426a1c
begin
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

Mapping back to generic terminology, **we simplify the learning problem by hard-coding part of it**. Importantly, we only know what to hard-code because we are experts. In this case, the authors "hard-coded" the 3D-to-2D part, since the part we really want to learn is just how to construct a 3D model from a bunch of 2D images.
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
\mathcal{O}\big(u(t,\mathbf{x}); \gamma \big) &= f\big(u(t,\mathbf{x})\big) \quad &&\mathbf{x}\text{ in } \Omega,\; t\in[0,T],\\
\mathcal{O}_\mathcal{B}\big(u(t,\mathbf{x})) &= f_B(u(t,\mathbf{x})) \quad &&\mathbf{x}\text{ in } \partial\Omega,\; t\in[0,T],\\
\end{align}
```

Here, $t$ is the temporal coordinate, $x$ is the spatial coordinate/state, and our neural network will learn to approximate the ground-truth solution $u(t,\mathbf{x})$.


Our goal is to fit a neural network $\hat{u}(t,\mathbf{x})$ that respects these conditions. In other words, we are solving: 

```math
\begin{align}
\text{find} &\enspace \theta \\
\text{such that} &\enspace
\mathcal{O}\big(\hat{u}_\theta(t,\mathbf{x}); \gamma\big) = f\big(\hat{u}_\theta(t,\mathbf{x})\big)  \quad \forall \mathbf{x} \in\Omega,\; t\in[0,T],\\
&\enspace \mathcal{O}_\mathcal{B}\big(\hat{u}_\theta(t,\mathbf{x})\big) = f_B\big(\hat{u}_\theta(t,\mathbf{x})\big)  \quad \forall \mathbf{x}\in\partial\Omega,\; t\in[0,T].
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


# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"

[compat]
PlutoUI = "~0.7.71"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.11.7"
manifest_format = "2.0"
project_hash = "0c76a76c3ac8f04e01e91e0dc955aee1f9d81e4a"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "6e1d2a35f2f90a4bc7c2ed98079b2ba09c35b83a"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.3.2"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.2"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"
version = "1.11.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "67e11ee83a43eb71ddc950302c53bf33f0690dfe"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.12.1"

    [deps.ColorTypes.extensions]
    StyledStringsExt = "StyledStrings"

    [deps.ColorTypes.weakdeps]
    StyledStrings = "f489334b-da3d-4c2e-b8f0-e476e12c162b"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.1.1+0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"
version = "1.11.0"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"
version = "1.11.0"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "05882d6995ae5c12bb5f36dd2ed3f61c98cbb172"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.5"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "179267cfa5e712760cd43dcae385d7ea90cc25a4"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.5"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "7134810b1afce04bbc1045ca1985fbe81ce17653"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.5"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "b6d6bfdd7ce25b0f9b2f6b3dd56b2673a66c8770"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.5"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
version = "1.11.0"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.4"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.6.0+0"

[[deps.LibGit2]]
deps = ["Base64", "LibGit2_jll", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"
version = "1.11.0"

[[deps.LibGit2_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll"]
uuid = "e37daf67-58a4-590a-8e99-b0245dd2ffc5"
version = "1.7.2+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.0+1"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"
version = "1.11.0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
version = "1.11.0"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"
version = "1.11.0"

[[deps.MIMEs]]
git-tree-sha1 = "c64d943587f7187e751162b3b84445bbbd79f691"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "1.1.0"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"
version = "1.11.0"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.6+0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"
version = "1.11.0"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2023.12.12"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.27+1"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "7d2f8f21da5db6a806faf7b9b292296da42b2810"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.3"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "Random", "SHA", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.11.0"

    [deps.Pkg.extensions]
    REPLExt = "REPL"

    [deps.Pkg.weakdeps]
    REPL = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "Downloads", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "8329a3a4f75e178c11c1ce2342778bcbbbfa7e3c"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.71"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "5aa36f7049a63a1528fe8f7c3f2113413ffd4e1f"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.2.1"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "0f27480397253da18fe2c12a4ba4eb9eb208bf3d"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.5.0"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"
version = "1.11.0"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
version = "1.11.0"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"
version = "1.11.0"

[[deps.Statistics]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "ae3bb1eb3bba077cd276bc5cfc337cc65c3075c0"
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.11.1"

    [deps.Statistics.extensions]
    SparseArraysExt = ["SparseArrays"]

    [deps.Statistics.weakdeps]
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
version = "1.11.0"

[[deps.Tricks]]
git-tree-sha1 = "372b90fe551c019541fafc6ff034199dc19c8436"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.12"

[[deps.URIs]]
git-tree-sha1 = "bef26fb046d031353ef97a82e3fdb6afe7f21b1a"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.6.1"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"
version = "1.11.0"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"
version = "1.11.0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+1"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.11.0+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.59.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+2"
"""

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
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
