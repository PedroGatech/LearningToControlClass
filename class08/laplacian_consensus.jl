### A Pluto.jl notebook ###
# v0.20.19

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    return quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

# ╔═╡ 462afff0-2ae3-4730-b48e-cf475fc9e14f
begin
	using Pkg
	Pkg.activate("class08/pluto_env")
	Pkg.instantiate()  # Installs packages from Manifest.toml if needed
	
	# Then import each package you want to use
	using Graphs
	using NetworkLayout
	using Plots
	using LinearAlgebra
	using PlutoUI
	using Statistics
	using GraphRecipes
	using Colors
	using PlutoTeachingTools
end

# ╔═╡ 75bdf059-c8ac-4f9c-b023-3c010b4389cb
md"""
# Consensus, ADMM, and Distributed Optimal Control

## Learning Objectives

* Understand the intuition behind consensus and why it arises in distributed settings
* Learn ADMM and how it enables distributed optimization
* Explore applications to distributed control and model predictive control (MPC)

## Table of Contents

1. Centralized vs. Distributed Control
2. Consensus Algorithms
3. ADMM for Consensus Optimization
4. Distributed Control Application
5. Connections with Other Lectures
6. Summary and Discussion

"""

# ╔═╡ fe6b8381-edeb-4d0c-87f5-4ecd2b7b9183
md"""
## 1. Centralized vs. Distributed Control

Consider a cooperative control setting involving a network of autonomous vehicles.  
Each agent $i$ is described by a simple double-integrator model:

```math
\dot{x}_i = v_i, \quad \dot{v}_i = u_i.
```

Suppose a centralized controller aims to maintain a desired formation or platoon.
__Since each agent’s behavior depends on its neighbors’ states__, the overall coordination problem can be expressed as:

```math
\min_{u} \sum_{i\in V} \int_{0}^{T} \left( ||x_i - x_i^{\text{ref}}||^2 + ||u_i||^2 + \alpha\sum_{j \in \mathcal{N}_i}||x_i - x_j||^2 \right) dt
```

where $\mathcal{N}_i$ denotes the neighbors of agent $i$ within the communication graph $G = (V,E)$.

While centralized control offers optimal coordination, it is often __impractical__ for large-scale systems due to limitations in scalability, privacy, and robustness.

A natural alternative is distributed control, where each agent solves a local optimization problem using only information from its immediate neighbors. This distributed formulation retains cooperative behavior while reducing computational and communication burdens:
```math
\min_{u_i} \int_{0}^{T} \left(||x_i - x_i^{\text{ref}}||^2 + ||u_i||^2 + \alpha\sum_{j \in \mathcal{N}_i}||x_i - x_j||^2 \right) dt
```
Here, each agent independently computes its control input $u_i$ and exchanges limited information with neighbors, and collectively the network reaches a coordinated outcome.
"""

# ╔═╡ 44154c9f-8e5e-49a7-b6ec-f465e0769c88
md"""
## 2. Consensus Algorithms

So what exactly is __consensus__?
It refers to _reaching an agreement on a certain quantity of interest that depends on the states of all agents in a network._

A __consensus algorithm__ is a protocol that enables this agreement through _local communication_, by exchanging information only with neighboring agents.

### 2A. Applications of Consensus
In the earlier example of distributed control for autonomous vehicles, each agent must achieve consensus over shared variables such as _relative positions_ and _velocities_.
This coordination behavior is commonly known as the __flocking problem__.

"""

# ╔═╡ 242f3a4c-ad0b-4619-80e6-1f1ee244b6f4
question_box(md"""This is just one example where consensus appears. Other examples include:
			 1. space rendezvous
			 2. power grid stability
			 3. federated learning
			 4. blockchain (distributed ledgers)
			 What shared variables require consensus in each of these settings?""")

# ╔═╡ a6c12abf-4303-45bc-99a7-7b55061013d6
Foldable(md"Answer...", md"""

		 1. space rendezvous - shared position
		 2. power grid stability - shared voltage
		 3. federated learning - shared global model
		 4. blockchain - shared transaction history
		 """)

# ╔═╡ cc94ceb2-e01a-4f10-9ee0-d0c906e5f46f
md"""
### 2B. Consensus Notation

We consider a network of decision-making agents with dynamics
$\dot{x}_i = u_i$,
where each agent communicates locally with its neighbors on a graph $G = (V, E)$, and with $|E| = N$ agents total.
The objective is to achieve asymptotic convergence to an agreement space such that
$x_1 = x_2 = \dots = x_n$.
Equivalently, this can be written as
$x = \alpha \mathbf{1}$,
where $\alpha \in \mathbb{R}$ represents the collective decision.

Let the adjacency matrix be $A = [a_{ij}]$.
The set of neighbors of agent $i$ is denoted by
$\mathcal{N}i = {, j \in V : a{ij} \neq 0 ,}$,
meaning that agent $i$ can communicate with agent $j$ whenever $a_{ij} \neq 0$.

In more general settings—such as mobile sensor networks or flocking—the graph can vary with time.
In this case, we write
$G(t) = (V, E(t))$, $A(t)$, and $\mathcal{N}_i(t)$
to denote the time-varying topology, adjacency matrix, and neighbor set, respectively.
"""

# ╔═╡ 453fb681-6f7b-42f2-9d10-0da7ad5b811c
md"""
### 2C. Distributed Consensus Algorithm
The distributed consensus algorithm takes the general form:
```math
\dot x_i(t) = \sum_{j \in \mathcal{N_i}} \left( x_j(t) - x_i(t) \right) + b_i(t)
```
where $b$ is an input bias, typically set to zero. A nonzero bias may represent, for instance, a desired inter-vehicle relative-position vector.

#### Average-Consensus
A common and instructive special case is the __average-consensus__ algorithm:
```math
\dot x_i(t) = \sum_{j \in \mathcal{N_i}} a_{ij} \left( x_j(t) - x_i(t) \right)
```
For graphs that are __connected__ and __undirected__ ($a_{ij} = a_{ji}$), the system satisfies a key invariance property: $\sum_i \dot x_i = 0$. This implies that the final consensus value must equal the average of the initial conditions:
```math
\alpha = \frac{1}{n}\sum_i x_i(0)
```
Hence, the algorithm asymptotically drives all agents to this average value for any initial state.
Average consensus finds wide application in sensor fusion, distributed estimation, and cooperative control problems within multi-agent systems.
"""

# ╔═╡ ee0fbf98-db44-4188-b623-3868a08c02b2
md"""
#### Graph Laplacian
The __Laplacian__ is a matrix defined as 
```math
L = [l_{ij}], \qquad
l_{ij} =
\begin{cases}
-1, & j \in \mathcal{N}_i, \\
|\mathcal{N}_i|, & j = i.
\end{cases}
```
The dynamics of the average-consensus algorithm can be compactly expressed as
```math
\dot x = -Lx
```
In an undirected graph, the matrix satisfies the sum-of-squares (SOS) property:
```math
x^\top Lx = \frac{1}{2}\sum_{(i,j)\in E}a_{ij}(x_j - x_i)^2
```
Defining the quadratic disagreement function $\varphi(x) = \frac{1}{2} x^T L x$, this yields the __gradient-descent__ algorithm:
```math
\dot{x} = -\nabla \varphi(x)
```
"""

# ╔═╡ 46a8121f-f1aa-4d22-8ef9-1d02f957767d
question_box(md"""What motivates the division by 2 in the quadratic disagreement funcion?""")

# ╔═╡ 85b74a70-a8df-4741-aa1b-d551d0e9bea2
Foldable(md"Answer...", md"It accounts for the double-counting of undirected edges in the graph.")

# ╔═╡ 524365a7-e799-4c55-acf4-88fcd5a716e2
md"""
### 2D. Demo: Connectivity and Consensus Performance
Spectral properties of $L$ affect the convergence of consensus algorithms. Try the demo below to see how __algebraic connectivity__ affects the performance in consensus algorithms.
"""

# ╔═╡ 3c5d6e70-6f43-11ef-3456-0123456789ab
md"""
##### Parameters

Adjust the parameters below to explore different graph topologies and consensus dynamics:

**Graph Structure:**

Number of nodes: $(@bind n_nodes Slider(5:50, default=15, show_value=true))

Topology: $(@bind topology Select(["random" => "Random (Erdős-Rényi)", "ring" => "Ring", "star" => "Star", "complete" => "Complete", "path" => "Path", "wheel" => "Wheel"]))

Connectivity (for random graphs): $(@bind connectivity Slider(0.1:0.05:0.9, default=0.3, show_value=true))

---

**Consensus Dynamics:**

Step size (α): $(@bind step_size Slider(0.01:0.01:0.5, default=0.1, show_value=true))

Number of iterations: $(@bind n_iterations Slider(10:10:200, default=100, show_value=true))
"""

# ╔═╡ 4d6e7f80-6f43-11ef-4567-0123456789ab
md"""
---
##### Graph Generation
"""

# ╔═╡ 5e7f8090-6f43-11ef-5678-0123456789ab
function create_graph(n::Int, topology_type::String, p::Float64=0.3)
	if topology_type == "complete"
		return complete_graph(n)
	elseif topology_type == "ring"
		return cycle_graph(n)
	elseif topology_type == "star"
		return star_graph(n)
	elseif topology_type == "grid"
		k = Int(ceil(sqrt(n)))
		# Create grid and take first n nodes
		g = grid([k, k])
		if nv(g) > n
			# Remove extra nodes
			for _ in 1:(nv(g) - n)
				rem_vertex!(g, nv(g))
			end
		end
		return g
	elseif topology_type == "path"
		return path_graph(n)
	elseif topology_type == "wheel"
		return wheel_graph(n)
	else  # random (Erdős-Rényi)
		return erdos_renyi(n, p)
	end
end

# ╔═╡ 6f809fa0-6f43-11ef-6789-0123456789ab
G = create_graph(n_nodes, topology, connectivity)

# ╔═╡ 7091a0b0-6f43-11ef-789a-0123456789ab
md"""
**Graph properties:**
- Nodes: $(nv(G))
- Edges: $(ne(G))
- Average degree: $(round(2 * ne(G) / nv(G), digits=2))
- Is connected: $(is_connected(G))
"""

# ╔═╡ 81a2b1c0-6f43-11ef-89ab-0123456789ab
md"""
---
##### Laplacian Matrix
"""

# ╔═╡ 92b3c2d0-6f43-11ef-9abc-0123456789ab
L = Matrix(laplacian_matrix(G))

# ╔═╡ a3c4d3e0-6f43-11ef-abcd-0123456789ab
md"""
Show Laplacian matrix: $(@bind show_laplacian CheckBox())
"""

# ╔═╡ b4d5e4f0-6f43-11ef-bcde-0123456789ab
if show_laplacian
	L
end

# ╔═╡ c5e6f500-6f43-11ef-cdef-0123456789ab
md"""
**Laplacian eigenvalues:**

The second smallest eigenvalue (algebraic connectivity) indicates how well-connected the graph is.
"""

# ╔═╡ d6f70610-6f43-11ef-def0-0123456789ab
begin
	λ = sort(eigvals(L))
	md"""
	- Smallest eigenvalue: $(round(λ[1], digits=6)) (should be ≈ 0)
	- Second smallest (algebraic connectivity): $(round(λ[2], digits=4))
	- Largest eigenvalue: $(round(λ[end], digits=4))
	"""
end

# ╔═╡ e7081720-6f43-11ef-ef01-0123456789ab
md"""
---
##### Initial Conditions
"""

# ╔═╡ f8192830-6f43-11ef-f012-0123456789ab
# Random initial values between 0 and 100
x₀ = rand(nv(G)) .* 100

# ╔═╡ 09293940-6f43-11ef-0123-0123456789ab
consensus_value = mean(x₀)

# ╔═╡ 1a3a4a50-6f43-11ef-1234-0123456789ab
md"""
**Initial statistics:**
- Mean (consensus value): $(round(consensus_value, digits=2))
- Std deviation: $(round(std(x₀), digits=2))
- Min: $(round(minimum(x₀), digits=2))
- Max: $(round(maximum(x₀), digits=2))
"""

# ╔═╡ 2b4b5b60-6f43-11ef-2345-0123456789ab
md"""
---
##### Consensus Dynamics
"""

# ╔═╡ 3c5c6c70-6f43-11ef-3456-0123456789ab
function consensus_evolution(L::Matrix, x₀::Vector, α::Float64, steps::Int)
	n = length(x₀)
	X = zeros(n, steps + 1)
	X[:, 1] = x₀
	
	for t in 1:steps
		# Discrete-time consensus update: x(t+1) = x(t) - α * L * x(t)
		X[:, t + 1] = X[:, t] - α * L * X[:, t]
	end
	
	return X
end

# ╔═╡ 4d6d7d80-6f43-11ef-4567-0123456789ab
X = consensus_evolution(L, x₀, step_size, n_iterations)

# ╔═╡ 6f8f9fa0-6f43-11ef-6789-0123456789ab
md"""
---
##### Visualization
"""

# ╔═╡ 92b2c2d0-6f43-11ef-9abc-0123456789ab
md"""
**Controls:**

Time step: $(@bind time_step Slider(0:n_iterations, default=0, show_value=true))

Node size: $(@bind node_size Slider(0.5:0.5:3.0, default=1.5, show_value=true))

Method: $(@bind method Select(["spring" => "Spring", "sfdp" => "SFDP", "stress" => "Stress"]))
"""

# ╔═╡ 5e7e8e90-6f43-11ef-5678-0123456789ab
md"""
**Statistics at time step $(time_step):**
- Mean: $(round(mean(X[:, time_step + 1]), digits=2))
- Variance: $(round(var(X[:, time_step + 1]), digits=4))
- Std deviation: $(round(std(X[:, time_step + 1]), digits=4))
- Distance to consensus: $(round(std(X[:, time_step + 1]), digits=4))
"""

# ╔═╡ a3c3d3e0-6f43-11ef-abcd-0123456789ab
begin
	# Get node colors based on current values
	node_values = X[:, time_step + 1]
	
	# Use the initial current min/max
	min_val_current = minimum(X[:, 1])
	max_val_current = maximum(X[:, 1])
	val_range = max_val_current - min_val_current
	
	# Create color palette from blue to red
	if val_range < 1e-6  # All values converged
		node_colors = [:purple for _ in 1:length(node_values)]
	else
		# Normalize to [0, 1]
		normalized_values = (node_values .- min_val_current) ./ val_range
		# Create colormap from blue (low) to red (high)
		node_colors = [RGB(v, 0.3*(1-v), 1-v) for v in normalized_values]
	end
	
	# Get layout based on method
	if method == "spring"
		layout_fn = spring
	elseif method == "sfdp"
		layout_fn = sfdp
	else
		layout_fn = stress
	end
	
	# Compute layout
	adj_mat = adjacency_matrix(G)
	pos = layout_fn(adj_mat)
	
	# Extract x and y coordinates
	xs = [p[1] for p in pos]
	ys = [p[2] for p in pos]
	
	# Create the plot
	p1 = plot(size=(700, 600), 
		      legend=false, 
		      framestyle=:none,
		      aspect_ratio=:equal,
		      title="Consensus State at t = $(time_step)\nConsensus: $(round(consensus_value, digits=2)) | Std: $(round(std(node_values), digits=3))")
	
	# Draw edges
	for e in edges(G)
		i, j = src(e), dst(e)
		plot!(p1, [xs[i], xs[j]], [ys[i], ys[j]], 
			  color=:gray, alpha=0.4, linewidth=1.5)
	end
	
	# Draw nodes
	scatter!(p1, xs, ys, 
		     markercolor=node_colors,
		     markersize=10*node_size,
		     markerstrokewidth=2,
		     markerstrokecolor=:black)
	
	p1
end

# ╔═╡ b4d4e4f0-6f43-11ef-bcde-0123456789ab
md"""
---
##### Node Value Evolution Over Time
"""

# ╔═╡ c5e5f500-6f43-11ef-cdef-0123456789ab
begin
	p2 = plot(0:n_iterations, X',
		alpha = 0.6,
		legend = false,
		xlabel = "Time step",
		ylabel = "Node value",
		title = "Consensus Convergence",
		linewidth = 2,
		size = (700, 400),
		margin = 5Plots.mm
	)
	
	# Add consensus line
	hline!(p2, [consensus_value],
		color = :red,
		linewidth = 3,
		linestyle = :dash,
		label = "Consensus value",
		legend = :topright
	)
	
	# Add current time marker
	vline!(p2, [time_step],
		color = :green,
		linewidth = 2,
		linestyle = :dot,
		alpha = 0.5
	)
end

# ╔═╡ d6f60610-6f43-11ef-def0-0123456789ab
md"""
---
##### Variance Over Time

"""

# ╔═╡ e7071720-6f43-11ef-ef01-0123456789ab
begin
	variances = [var(X[:, t]) for t in 1:(n_iterations+1)]
	
	p3 = plot(0:n_iterations, variances,
		xlabel = "Time step",
		ylabel = "Variance",
		title = "Convergence to Consensus",
		linewidth = 2,
		color = :purple,
		legend = false,
		size = (700, 400),
		margin = 5Plots.mm,
		yscale = :log10
	)
	
	vline!(p3, [time_step],
		color = :green,
		linewidth = 2,
		linestyle = :dot,
		alpha = 0.5
	)
end

# ╔═╡ f8182830-6f43-11ef-f012-0123456789ab
md"""
---
##### Analysis
A larger algebraic connectivity means faster convergence. This depends largely on the chosen topology. __Step size__ also affects stability.
- Too small: Slow convergence
- Too large: May cause oscillations or instability
- Optimal: Related to the largest eigenvalue of L (α < 2/λ_max for stability)

**Maximum safe step size for current graph:** α < $(round(2/maximum(λ), digits=3))
"""

# ╔═╡ 7e2909e5-0667-4589-b3df-24f48ae67fd8
md"""
## 3. ADMM for Consensus
The __Alternating Direction Method of Multipliers (ADMM)__, first introduced in the 1970s, is a powerful decomposition–coordination algorithm designed to solve large optimization problems by breaking them into smaller subproblems that can be solved in parallel and then coordinated toward a global solution.

To understand how ADMM applies to __consensus__, we first review several precursor methods that motivate its structure and intuition.

### 3A. Dual Ascent Method
We begin with the **convex optimization problem**:

```math
\begin{aligned}
&\text{minimize} && f(x) \quad \quad \quad \quad (1) \\
&\text{subject to} && Ax = b
\end{aligned}
```
The __Lagrangian__ is defined as:
```math
L(x,y) = f(x) + y^\top (Ax - b)
```
The __dual function__ is obtained by minimizing the Lagrangian over $x$:
```math
g(y) = \inf_{x} L(x,y) = -f^*(-A^\top y) - b^\top y
```
and the __dual problem__ seeks to maximize $g(y)$, where $f^*$ represents the convex conjugate of $f$.

The dual ascent iterations alternate between minimizing the Lagrangian over $x$ and updating the dual variable $y$:
```math
\begin{aligned}
x^{k+1} &:= \arg\min_x L(x, y^k) \\
y^{k+1} &:= y^k + \alpha^k (A x^{k+1} - b)
\end{aligned}
```
where __residual__ $(A x^{k+1} - b)$ acts as a gradient (or subgradient) direction.

However, this method comes with restrictive convergence assumptions: $f$ must be strictly convex, and $L$ must remain bounded in $x$ for every $y$. As a result, it is not widely applicable in practice.
"""

# ╔═╡ 8bcd8d66-fc1c-419f-8e36-374c2c22965a
md"""
### 3B. Dual Decomposition
The next precursor method is __dual decomposition__. Suppose now that the objective is separable: 
```math
f(x) = \sum_{i = 1}^N f_i(x_i)
```
The $x$-minimization step now splits into $N$ separate problems which can now be solved in parallel, reducing computation time and improving scalability.
```math
\begin{aligned}
	x_i^{k+1} &:= \arg\min_{x_i} L_i(x_i, y^{k}) \\
	y^{k+1} &:= y^{k} + \alpha^{k} (A x^{k+1} - b)
\end{aligned}
```
"""

# ╔═╡ f75ec743-d448-432f-9511-b2c7a382873c
md"""
### 3C. Augmented Lagrangians and Method of Multipliers
The final precursor method is the __Augmented Lagrangian__ and __Method of Multipliers__.

The __augmented Lagrangian__ adds an additional quadratic penalty term:
```math
L_{\rho}(x, y) = f(x) + y^{T}(A x - b) + \frac{\rho}{2} \|A x - b\|_{2}^{2}
```
It corresponds with the problem
```math
\begin{aligned}
        &\text{minimize} \quad f(x) + \frac{\rho}{2}\|A x - b\|_{2}^{2} \\
        &\text{subject to} \quad A x = b
        \end{aligned}
```
This is clearly equivalent with Model (1), since the penalty is zero for any feasibly $x$. 

Applying the dual ascent algorithm yields the algorithm known as the __method of multipliers__:
```math
\begin{aligned}
	x^{k+1} &:= \arg\min_{x} L_{\rho}(x, y^{k})\\
	y^{k+1} &:= y^{k} + \rho (A x^{k+1} - b)
\end{aligned}
```
This method improves robustness of the dual ascent algorithm, no longer requiring the assumptions of strict convexity or finiteness of $f$. However, $f$ must still be convex and closed. Slater's condition is still needed to guarantee strong duality.

__Note that penalty $\rho$ is chosen as the step-size of the algorithm now__: the quadratic penalty makes the dual smoother but "scales" it's gradient; this is compensated by taking step-sizes of $\rho$.

Finally, note that even when $f$ is separable, the augmented Lagrangian $L_{\rho}$ is not!!! 
"""

# ╔═╡ Cell order:
# ╠═462afff0-2ae3-4730-b48e-cf475fc9e14f
# ╟─75bdf059-c8ac-4f9c-b023-3c010b4389cb
# ╟─fe6b8381-edeb-4d0c-87f5-4ecd2b7b9183
# ╟─44154c9f-8e5e-49a7-b6ec-f465e0769c88
# ╟─242f3a4c-ad0b-4619-80e6-1f1ee244b6f4
# ╟─a6c12abf-4303-45bc-99a7-7b55061013d6
# ╠═cc94ceb2-e01a-4f10-9ee0-d0c906e5f46f
# ╟─453fb681-6f7b-42f2-9d10-0da7ad5b811c
# ╟─ee0fbf98-db44-4188-b623-3868a08c02b2
# ╟─46a8121f-f1aa-4d22-8ef9-1d02f957767d
# ╟─85b74a70-a8df-4741-aa1b-d551d0e9bea2
# ╠═524365a7-e799-4c55-acf4-88fcd5a716e2
# ╟─3c5d6e70-6f43-11ef-3456-0123456789ab
# ╟─4d6e7f80-6f43-11ef-4567-0123456789ab
# ╟─5e7f8090-6f43-11ef-5678-0123456789ab
# ╟─6f809fa0-6f43-11ef-6789-0123456789ab
# ╟─7091a0b0-6f43-11ef-789a-0123456789ab
# ╟─81a2b1c0-6f43-11ef-89ab-0123456789ab
# ╟─92b3c2d0-6f43-11ef-9abc-0123456789ab
# ╟─a3c4d3e0-6f43-11ef-abcd-0123456789ab
# ╟─b4d5e4f0-6f43-11ef-bcde-0123456789ab
# ╟─c5e6f500-6f43-11ef-cdef-0123456789ab
# ╟─d6f70610-6f43-11ef-def0-0123456789ab
# ╟─e7081720-6f43-11ef-ef01-0123456789ab
# ╟─f8192830-6f43-11ef-f012-0123456789ab
# ╟─09293940-6f43-11ef-0123-0123456789ab
# ╟─1a3a4a50-6f43-11ef-1234-0123456789ab
# ╟─2b4b5b60-6f43-11ef-2345-0123456789ab
# ╟─3c5c6c70-6f43-11ef-3456-0123456789ab
# ╟─4d6d7d80-6f43-11ef-4567-0123456789ab
# ╟─5e7e8e90-6f43-11ef-5678-0123456789ab
# ╟─6f8f9fa0-6f43-11ef-6789-0123456789ab
# ╟─92b2c2d0-6f43-11ef-9abc-0123456789ab
# ╟─a3c3d3e0-6f43-11ef-abcd-0123456789ab
# ╟─b4d4e4f0-6f43-11ef-bcde-0123456789ab
# ╟─c5e5f500-6f43-11ef-cdef-0123456789ab
# ╟─d6f60610-6f43-11ef-def0-0123456789ab
# ╟─e7071720-6f43-11ef-ef01-0123456789ab
# ╟─f8182830-6f43-11ef-f012-0123456789ab
# ╟─7e2909e5-0667-4589-b3df-24f48ae67fd8
# ╟─8bcd8d66-fc1c-419f-8e36-374c2c22965a
# ╠═f75ec743-d448-432f-9511-b2c7a382873c
