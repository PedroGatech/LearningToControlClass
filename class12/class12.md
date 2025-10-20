# Class 12 — 11/07/2025

**Presenter:** Pedro Paulo

**Topic:** Neural operators (FNO, Galerkin Transformer); large-scale surrogates

# Foundations of Neural Operators

## Notations and definitions
Let's start by setting up some notations:
- Derivatives $\dfrac{\partial}{\partial t}$ are going to be written as $\partial_t$, and, in the case of derivatives **w.r.t. time**, they could be written as $\partial_t x=\dot x$.
- Integrals $\int \{\cdot\} \ \mathrm dt$ are going to be written as $\int \mathrm dt \ \{\cdot\}$.

And having some definitions:
- **Vectors** are *lists of numbers*, i.e., a vector $v$ lives in $\mathbb R^{d_v}$, and can be thought as a list of $d_v$ numbers, all in $\mathbb R$. More generally, vectors could live in a generic *vector space* $V$, so we would have $v\in V$. 
- **Functions** are vector-to-vector mapping, i.e., a function $f$ brings a $v \in \mathbb R^{d_v}$ to a $w \in \mathbb R^{d_w}$, and we define that as $f: \mathbb R^{d_v} \rightarrow \mathbb R^{d_w}$.  More generally, functions could operate on a generic *vector space* $V$ and $W$, so we would have $f: V \rightarrow W$. 
- **Operators** are function-to-functions mapping, i.e., an operator $A$ brings an $f:\mathbb R^{d_{v1}} \rightarrow \mathbb R^{d_{w1}}$ to a $g: \mathbb R^{d_{v2}} \rightarrow \mathbb R^{d_{w2}}$. More generally, operators could operate on generic *function spaces*, so we would have an operator $A$ bringing an $f:V_1 \rightarrow W_1$ to a $g:V_2 \rightarrow W_2$. 

Key differences:
- A vector is *naturally* discrete. Therefore, the input-output pair for functions are also *naturally* discrete. 
- A function is *naturally* continuous. Therefore, the input-output pair for operators are also *naturally* continuous.

It is said that Neural Networks (NN) are **universal function approximators** \[cite], in this section we're going to ~~try to~~ create the idea of **universal operator approximators**, that map functions to functions, using something called **Neural Operators**.

A NN $\mathcal N$ can be thought as a general **function** $\mathcal N: X \times \Theta \rightarrow Y$, where $X$ and $Y$ are vector spaces, and $\Theta$ is the parameter space. So we take elements $x \in X$ and we *learn* how to map those onto $y\in Y$, by means of changing the parameters $\theta \in \Theta$. That way, we can approximate any function (that's where the "universal function approximator" comes from) that maps $X \rightarrow Y$. 
In a similar way we can think about a Neural Operator $\mathcal G^\dagger: \mathcal X \times \Theta \rightarrow \mathcal Y$, where $\mathcal X$ and $\mathcal Y$ are function spaces, and $\Theta$ is the parameter space. Now, instead of learning how to map *vectors*, we're going to learn the mapping of *functions*. This general idea will be expanded further.

**Why are functions important?** Everything in the real world is a function! If we want to predict the airflow around a car, the stress caused by deforming a metal bar, the temperature of a reactor, the weather (and the list goes on), we would need to use functions.
When putting into a computer we are going to need to mesh our function, otherwise we'd not be able to process it. But we're going to think about functions when designing the architecture of these Neural Operators.

**Why approximate operators?** Let's start with a parallel with image processing. Imagine that I have a Convolutional NN (CNN) that take as an input a (discrete) $256\times256$ image (let's imagine it in grayscale for simplicity). The input to this CNN would then be a $v \in \mathbb R^{256 \times 256}$, where each element $v_i \in \mathbb R \ ; v_i \in [0,1]$. Although this is a typical architecture for image processing \[cite], and it has been around since _year_ \[cite], it has a couple of limitations:
- The input **has to** be $256\times256$, the need of different dimension leads to a new NN and a new training.
- In case of regression, the output **has to** a fixed dimension, the need of different dimension leads to a new NN and a new training.
For the case of image processing, where there's no trivial underlying function behind the image, we cannot take advantage of the use of Neural Operators, but in the case of distributions of physical quantities, e.g., temperature, where there's a underlying function behind it, we can leverage the use of Neural Operators to understand distribution function, and make predictions/controls based on it, decoupling the parametrization $\Theta$ from the discretization of the data. \[cite] *et al.* compared the errors of two networks: U-Net (NN topology) and PCA-Net (Neural operator topology), that were trained on different discretizations of the *same underlying function*, and the result is shown below:

This brings a concept (that we'll try to keep with our definition of Neural Operators) called **Discretization Invariance**:
- When we have Discretization Invariance we de-couple the parameters and the cost from the discretization, i.e., when changing the discretization the error doesn't vary.
- If our model is Discretization Invariable, we can use information at different discretizations to train, and we can transfer parameters learned for one discretization to another, that leads to something called "zero-shot super-resolution", that basically consists of training into a smaller discretization and predicting into a bigger one, due to the Discretization Invariance. This concept, together with its limitations, will be discussed in the "Fourier Neural Operator" section.
 
# Operator basics
Let the operator $\mathcal G: \mathcal X \rightarrow \mathcal Y$, where $\mathcal X$ and  are separable Banach spaces (mathematical way of saying that $\mathcal X$ and $\mathcal Y$ are spaces of functions) of vector-valued functions:
$$
\begin{flalign}
\mathcal X&=\{x: D\rightarrow \mathbb R\}, \ \ \ D \subseteq\mathbb R^d
\\
\mathcal Y&=\{y: D\rightarrow \mathbb R\}
\end{flalign}
$$
For example, $D$ is a cut plane of a biological tissue ($D \subseteq\mathbb R^2$) under the application of electric fields, and $x\in\mathcal X$ and $y\in\mathcal Y$ are temperatures before and after the application of said fields. The operator $\mathcal G$ is given by:
$$
\rho c_p\partial_tT =\nabla \cdot(k\nabla T) + \sigma|E|^2-Q
$$
where
$$
\begin{flalign}
\rho &\text{ is the tissue's density}
\\
c_p &\text{ is the tissue's heat capacity}
\\
T &\text{ is the temperature distribution on the tissue}
\\
k &\text{ is the tissue's thermal conductivity}
\\
\sigma &\text{ is the tissue's electrical conductivity}
\\
E &\text{ is the electric field distribution}
\\
Q &\text{ is the heat transfer, from blood/metabolism}
\end{flalign}
$$
This is one specific case of an operator, but any PDE can be thought as an operator.

## Approximations

Imagine that I want to approximate this operator $\mathcal G$ by means of an $\mathcal G^\dagger: \mathcal X\times\Theta \rightarrow \mathcal Y$, a first idea could be to find two linear mappings, here called $K_\mathcal W$ and $L_\mathcal W$ such that $K_\mathcal WL_\mathcal W \approx I$, where $I$ is the identity operator (i.e., by applying $K_\mathcal W$ and $L_\mathcal W$ to *all* $w \in \mathcal W$ we will return to the same $w$), and that by applying $K_\mathcal W$ to a $w\in\mathcal W$ we can project this $w$ onto a non-infinite space $\mathbb R^{d_\mathcal W}$ (one example of such operator is the FFT family, we're we approximate every function to a finite set of coefficients that represent the original functions' sines and cosines). By doing this to both $\mathcal X$ and $\mathcal Y$, we're going to have two non-infinite representations of $\mathcal X$ and $\mathcal Y$, on $\mathbb R^{n}$ and $\mathbb R^{m}$, respectively, and we can map this two representations using a non-linear function $\varphi$. 

A general diagram is shown below:

In this case, we can see that our $\mathcal G^\dagger$ can be given by $\mathcal G^\dagger = K_\mathcal X \circ \varphi\circ L_\mathcal Y$, where $K_\mathcal X$ and $L_\mathcal Y$ are the operators that project $\mathcal X$ and $\mathcal Y$ to the non-infinite dimension spaces $\mathbb R^{n}$ and $\mathbb R^{n}$, respectively, and $\varphi$ is a non-linear function that maps $\mathbb R^{n}$ to $\mathbb R^{m}$. Different selections of the set {$K_\mathcal W$, $L_\mathcal W$, $\varphi$} generate different classes of Neural Operators.

We can, from this, see the first limitation of this technique: we're limited by how well is the approximation of $K_\mathcal WL_\mathcal W \approx I$. It turns out that, as described by \[cite], this is approximation is fairly general:
Universal approximation:
Let:
- $\mathcal X$ and $\mathcal Y$ be separable Banach spaces.
- $\mathcal G: \mathcal X \rightarrow \mathcal Y$ be continuous.
For any $U\subset \mathcal X$ compact and $\epsilon > 0$, *there exists* continuous, linear maps $K_\mathcal X:\mathcal X \rightarrow \mathbb R^n$,  $L_\mathcal Y:\mathcal Y \rightarrow \mathbb R^m$, and $\varphi: \mathbb R^n \rightarrow \mathbb R^m$ such that:
$$
\sup_{u\in U} \| \mathcal G(u)-\mathcal G^\dagger(u)\|_\mathcal Y < \epsilon
$$
Average approximation:
Let: 
- $\mathcal X$ be separable Banach spaces, and $\mu \in \mathcal P(\mathcal X)$ be a probability measure in $\mathcal X$.
- $\mathcal G \in L_\mu^p(\mathcal X;\mathcal Y)$ for some $1\leq p < \infty$
If $\mathcal Y$ is separable Hilbert space, and $\epsilon > 0$, *there exists* continuous, linear maps $K_\mathcal X:\mathcal X \rightarrow \mathbb R^n$,  $L_\mathcal Y:\mathcal Y \rightarrow \mathbb R^m$, and $\varphi: \mathbb R^n \rightarrow \mathbb R^m$ such that:
$$
\| \mathcal G(u)-\mathcal G^\dagger(u)\|_{L_\mu^p(\mathcal X;\mathcal Y)} < \epsilon
$$
Let's start by giving two classes of Neural Operators, the Principal Component Analysis Network (PCA-NET) and the Deep Operator Network (DeepONet).

## PCA
First proposed by \[cite], we're going to define the PCA-NET approximation by analyzing our input and output spaces using a PCA-like technique.
Let:
- $\mathcal X$ and $\mathcal Y$ be separable Banach spaces, and let $x\in K\subset\mathcal X$, with $K$ compact.
- $\mathcal G$ (the operator that we're trying to approximate) be continuous.
- $\varphi_j:\mathbb R^n \times \Theta \rightarrow \mathbb R^m$ be multiple neural networks.
- $\xi_1,\text{...},\xi_n$ be the PCA basis functions of the input space $\mathcal X$.
	- The operator $K_\mathcal X$ for a given $x\in \mathcal X$ would then be $K_\mathcal X(x) :=\mathrm Lx = \{\langle\xi_j,x\rangle\}_j$.
- $\psi_1,\text{...},\psi_m$ be the PCA basis functions of the output space $\mathcal Y$.

The final approximation $\mathcal G^\dagger_{\text{PCA}}:\mathcal X \times \Theta \rightarrow \mathcal Y$ is then given by:
```math
\begin{align}
\mathcal G^\dagger_{\text{PCA}}&(x;\theta)(u)=\sum_{j=0}^m\varphi_j(\mathrm Lx;\theta)\psi_j(u) \ \ \ \ \forall\ x\in\mathcal X  \ \ \ \ u\in D_u
\end{align}
```
That is, the output is the *linear combination* of the PCA output basis functions {$\psi_j$}, weighted by NN coefficients $\varphi_j$, that have as input the $\mathrm Lx$ mapping of the input to the PCA space.

## DeepONet
Proposed by \[cite], the DeepONet generalizes the idea of PCA-NET, by means of *learning* the PCA basis functions of the output space $\mathcal Y$, i.e., $\psi_1,...,\psi_m$ are now NNs. The parameter space is then composed of two distinct set of parameters to be learned: $\theta_\varphi$, the same parameters as the original PCA-NET, and $\theta_\psi$, the parameters for the PCA basis functions of the output space. We will then have:

```math
\begin{align}
\mathcal G^\dagger_{\text{DeepONet}}&(x;\theta)(u)=\sum_{j=0}^m\varphi_j(\mathrm Lx;\theta_\varphi)\psi_j(u;\theta_\psi) \ \ \ \ \forall\ x\in\mathcal X  \ \ \ \ u\in D_u
\end{align}
```

## Overcoming the curse of dimensionality
One of the big problems of these approaches is the fact $L_\mathcal Y$ is a linear combination of the {$\psi_j$}. This leads to the need of an doubly exponential growth in the amount of data, when compared to $n$ (the size of the PCA basis functions of the input space $\mathcal X$), to achieve convergence \[cite]. To overcome this difficulty, we're going to generalize this idea of linear approximation of operators to the non-linear case.

Let:
- $\mathcal X$ and $\mathcal Y$ be function spaces over $\Omega \subset \mathbb R^d$
- $\mathcal G^\dagger$ is the composition of non-linear operators: $\mathcal G^\dagger=S_1\circ \text{...} \circ S_L$ 
	- In the linear case, as described before, $S_1 = K_\mathcal X$, $S_L = K_\mathcal Y$ and they're connected through multiple $\varphi_j$.
The above definition *looks a lot* like the typical definition of NNs, where each one of the $S_l$ is a layer of your NN. And, as we're going to see, it is! At least it is a generalization of the definition of NN to function space.
\[cite] *et al.* proposed to create each one of this $S_l$ as follows:
$$
S_l(a)(x) = \sigma_l\bigg( W_la(x) + b_l + \int_\Omega\mathrm dz \ \kappa_l(x,z)a(z)  \bigg), \ \ \ \ x \in \Omega
$$
where:
- $\sigma_l:\mathbb R^k\rightarrow\mathbb R^k$ is the non-linear activation function.
- $W_l\in\mathbb R^k$ is a term related to a "residual network".
	- This term is not necessary for convergence, but it's credited to help with convergence speed \[cite].
- $b_l\in\mathbb R^k$ is the bias term.
- $\kappa_l:\Omega\times\Omega\rightarrow\mathbb R^k$ is the kernel function.

The main distinction between this approach and the traditional NN approach is the $\kappa_l$ term, instead of the traditional weights, and the fact that the input $a(x)$ is a *function*, instead of a vector like the traditional NNs.
Different selections of $\kappa_l$ generate different classes of these non-linear Neural Operators, but we're going to focus on the transform $\kappa_l$, more specifically the Fourier Neural Operator and the Garlekin Transformer.

# Fourier Neural Operator
Let $\kappa_l(x,z)=\kappa_l(x-z)$, the integral will then become:
$$
\int_\Omega \mathrm dz \ \kappa_l(x,z)a(z) = \int_\Omega \mathrm dz \ \kappa_l(x-z)a(z) =\kappa_l(x) * a(x)
$$
where $*$ represents the convolution operator.
And, as we know from Fourier Transformation Theory, 
$$
\mathcal F\{\kappa_l(x)*a(x)\} = \mathcal F\{\kappa_l(x)\} \cdot\mathcal F\{a(x)\}
$$
where $\mathcal F\{\cdot\}$ represents the Fourier transform of a function.
We can than reduce the single layer $S_l$ represented before to the following:
$$
S_l(a)(x) = \sigma_l\bigg( W_la(x) + b_l + \mathcal F^{-1}\{\mathcal F\{\kappa_l(x)\} \cdot\mathcal F\{a(x)\}\}  \bigg), \ \ \ \ x \in \Omega
$$
This is basically what defines the Fourier Neural operator: the Neural Operator $\mathcal G^\dagger=S_1\circ \text{...} \circ S_L$ where each one of these $S_l$ is done by up/downscaling the previous output function using its fourier expansions.



# Galerkin Transformer
-- TODO --
Papers to cite:

📖 S. Cao: Choose a Transformer: Fourier or Galerkin (Available [here](https://arxiv.org/abs/2105.14995))

📖 X. Wang: Exploring Efficient Partial Differential Equation Solution Using Speed Galerkin Transformer (Available [here](https://ieeexplore.ieee.org/abstract/document/10793230))

📖 H. Wu: Transolver: A Fast Transformer Solver for PDEs on General Geometries (Available [here](https://arxiv.org/pdf/2402.02366))


# Large-scale surrogates
-- TODO --
Papers to cite:

📖 T. Grady: Model-Parallel Fourier Neural Operators as Learned Surrogates for Large-Scale Parametric PDEs (Available [here](https://www.sciencedirect.com/science/article/pii/S0098300423001061?casa_token=49-AswW96sUAAAAA:rgUui8eHQVtqwTAn4uzR4-s9i5_ThGu0Fl3m_GI6i5xgYUMbHpgjwkJYgW9l6VFGPdCCjA_LUck))

📖 L. Meyer: Training Deep Surrogate Models with Large Scale Online Learning (Available [here](https://proceedings.mlr.press/v202/meyer23b/meyer23b.pdf))

