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

# ╔═╡ 8a2c3e40-6f42-11ef-1234-0123456789ab
begin
	using Pkg
	Pkg.activate("class08/pluto_env")
	Pkg.add(["Graphs", "NetworkLayout", "Plots", "LinearAlgebra", "PlutoUI", "Statistics", "GraphRecipes"])
	
	using Graphs
	using NetworkLayout
	using Plots
	using LinearAlgebra
	using PlutoUI
	using Statistics
	using GraphRecipes
	
	gr()  # Use GR backend
end

# ╔═╡ 75bdf059-c8ac-4f9c-b023-3c010b4389cb
md"""
# Consensus, ADMM, and Distributed Optimal Control

Before we dive into the discussion of the theory behind consensus and ADMM, we must properly motivate the methods.

## Why Distributed Optimal Control?

Distributed optimal control involves the """

# ╔═╡ 2b4c5d60-6f43-11ef-2345-0123456789ab
md"""
# Laplacian-Based Consensus Algorithm

This notebook implements a **consensus algorithm** using the graph Laplacian matrix. The consensus dynamics follow:

```math
\frac{dx}{dt} = -Lx
```

where ``L`` is the Laplacian matrix of the graph and ``x`` is the vector of node values.

## Theory

The Laplacian matrix ``L = D - A`` where:
- ``D`` is the degree matrix (diagonal)
- ``A`` is the adjacency matrix

For consensus, all nodes converge to the average of their initial values: ``\bar{x} = \frac{1}{n}\sum_{i=1}^{n} x_i(0)``
"""

# ╔═╡ 3c5d6e70-6f43-11ef-3456-0123456789ab
md"""
## Parameters

Adjust the parameters below to explore different graph topologies and consensus dynamics:

**Graph Structure:**

Number of nodes: $(@bind n_nodes Slider(5:50, default=15, show_value=true))

Topology: $(@bind topology Select(["random" => "Random (Erdős-Rényi)", "ring" => "Ring", "star" => "Star", "complete" => "Complete", "grid" => "Grid", "path" => "Path", "wheel" => "Wheel"]))

Connectivity (for random graphs): $(@bind connectivity Slider(0.1:0.05:0.9, default=0.3, show_value=true))

---

**Consensus Dynamics:**

Step size (α): $(@bind step_size Slider(0.01:0.01:0.5, default=0.1, show_value=true))

Number of iterations: $(@bind n_iterations Slider(10:10:200, default=100, show_value=true))
"""

# ╔═╡ 4d6e7f80-6f43-11ef-4567-0123456789ab
md"""
---
## Graph Generation
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
## Laplacian Matrix
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
## Initial Conditions
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
## Consensus Dynamics
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
## Visualization
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
	
	# Use the actual current min/max for better color contrast
	min_val_current = minimum(node_values)
	max_val_current = maximum(node_values)
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
### Node Value Evolution Over Time
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
### Variance Over Time

This plot shows how quickly the network converges to consensus.
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
## Analysis

### Convergence Rate

The convergence rate of the consensus algorithm depends on the **algebraic connectivity** (second smallest eigenvalue of the Laplacian). A larger algebraic connectivity means faster convergence.

### Topology Effects

Different topologies have different convergence properties:
- **Complete graph**: Fastest convergence (all nodes connected)
- **Star graph**: Central node acts as mediator
- **Ring graph**: Slower convergence (information propagates around the ring)
- **Grid graph**: Moderate convergence speed
- **Random graphs**: Depends on connectivity parameter

### Step Size

The step size α affects stability:
- Too small: Slow convergence
- Too large: May cause oscillations or instability
- Optimal: Related to the largest eigenvalue of L (α < 2/λ_max for stability)

**Maximum safe step size for current graph:** α < $(round(2/maximum(λ), digits=3))
"""

# ╔═╡ 09294940-6f43-11ef-0123-0123456789ab
md"""
---
## Try This

1. **Compare topologies**: Switch between different graph types and observe convergence speed
2. **Adjust connectivity**: For random graphs, see how connectivity affects convergence
3. **Step size exploration**: Try different step sizes and watch for instability (try α > 2/λ_max!)
4. **Initial conditions**: The final consensus value is always the mean of initial values!
5. **Algebraic connectivity**: Check the second eigenvalue for different topologies
6. **Watch the variance plot**: The rate of exponential decay depends on the graph structure

"""

# ╔═╡ Cell order:
# ╠═75bdf059-c8ac-4f9c-b023-3c010b4389cb
# ╟─2b4c5d60-6f43-11ef-2345-0123456789ab
# ╠═8a2c3e40-6f42-11ef-1234-0123456789ab
# ╟─3c5d6e70-6f43-11ef-3456-0123456789ab
# ╟─4d6e7f80-6f43-11ef-4567-0123456789ab
# ╠═5e7f8090-6f43-11ef-5678-0123456789ab
# ╠═6f809fa0-6f43-11ef-6789-0123456789ab
# ╟─7091a0b0-6f43-11ef-789a-0123456789ab
# ╟─81a2b1c0-6f43-11ef-89ab-0123456789ab
# ╠═92b3c2d0-6f43-11ef-9abc-0123456789ab
# ╟─a3c4d3e0-6f43-11ef-abcd-0123456789ab
# ╟─b4d5e4f0-6f43-11ef-bcde-0123456789ab
# ╟─c5e6f500-6f43-11ef-cdef-0123456789ab
# ╟─d6f70610-6f43-11ef-def0-0123456789ab
# ╟─e7081720-6f43-11ef-ef01-0123456789ab
# ╠═f8192830-6f43-11ef-f012-0123456789ab
# ╠═09293940-6f43-11ef-0123-0123456789ab
# ╟─1a3a4a50-6f43-11ef-1234-0123456789ab
# ╟─2b4b5b60-6f43-11ef-2345-0123456789ab
# ╠═3c5c6c70-6f43-11ef-3456-0123456789ab
# ╠═4d6d7d80-6f43-11ef-4567-0123456789ab
# ╟─5e7e8e90-6f43-11ef-5678-0123456789ab
# ╟─6f8f9fa0-6f43-11ef-6789-0123456789ab
# ╟─92b2c2d0-6f43-11ef-9abc-0123456789ab
# ╠═a3c3d3e0-6f43-11ef-abcd-0123456789ab
# ╟─b4d4e4f0-6f43-11ef-bcde-0123456789ab
# ╠═c5e5f500-6f43-11ef-cdef-0123456789ab
# ╟─d6f60610-6f43-11ef-def0-0123456789ab
# ╠═e7071720-6f43-11ef-ef01-0123456789ab
# ╟─f8182830-6f43-11ef-f012-0123456789ab
# ╟─09294940-6f43-11ef-0123-0123456789ab
