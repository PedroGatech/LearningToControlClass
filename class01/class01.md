# Class 1 — 08/22/2025

**Presenter:** Andrew Rosemberg

**Topic:** Course map; why PDE-constrained optimization; tooling overview; stability & state-space dynamics; Lyapunov; discretization issues.

---

In this first class, we will introduce the course, outlining the topics that will be covered and the importance of PDE-constrained optimization. We discuss the tools that will be used throughout the course and provides an overview of stability and state-space dynamics, including Lyapunov stability. The class also touches on discretization issues that arise in the context of PDEs.

## Pre-requisites

- **Mathematical Background:** A solid understanding of calculus and linear algebra is essential. Familiarity with differential equations and optimization techniques is also needed.

- **Programming Skills:** Intermediate capacity in a programming language such as Julia, Python, or MATLAB is required for implementing the concepts discussed in class.

## Background Material

A few background materials have been selected to help you prepare for the course. Please review these resources before the first class:

### **Git**: 
Familiarity with Git for version control is recommended. Students should be comfortable with basic Git commands and workflows. Go through the [Git Basics](./background_materials/git_adventure_guide.md) guide to get started.

### **Julia and Pluto**: 
The course will use Julia for programming assignments and projects. If you are new to Julia, go through either or all of these resources until you feel comfortable with the language:

- [Julia for Beginners](https://juliaacademy.com/p/julia-for-beginners)
- [Parallel Computing and Scientific Machine Learning (SciML): Methods and Applications](https://book.sciml.ai/)
- [JuMP Julia Tutorial](https://jump.dev/JuMP.jl/stable/tutorials/getting_started/getting_started_with_julia/)
- [Julia ML Course](https://adrianhill.de/julia-ml-course/)

Julia is a high-level, general-purpose dynamic programming language, designed to be fast and productive, for e.g. data science, artificial intelligence, machine learning, modeling and simulation, most commonly used for numerical analysis and computational science.

Pluto is a programming environment for _Julia_, designed to be **interactive** and **helpful**. Like Jupyter but with more features. [Some other Example Notebooks](https://featured.plutojl.org/)

A few steps to get started with Julia and Pluto:

#### Step 1: Install Julia 

Go to https://julialang.org/downloads and download the current stable release, using the correct version for your operating system (Linux x86, Mac, Windows, etc).

#### Step 2: Run Julia

After installing, make sure that you can run Julia.

#### Step 3: Install Pluto and other dependencies

Just activate and instantiate the project environment (provided in the class folder):

```julia
using Pkg
Pkg.activate("path/to/Class01_Folder")
```

`Project.toml` and `Manifest.toml` files are provided in the class folder, which will install all the necessary packages for this course at the correct versions.

#### Step 4: Start Pluto

```julia
julia> using Pluto

julia> Pluto.run()
```

#### Step 5: Opening an existing notebook file

To run a local notebook file that you have not opened before, then you need to enter its full path (e.g. `path/to/math_basics.jl`) into the blue box in the main menu.

### **Linear Algebra**: 
We have prepared a basic (Pluto) [Linear Algebra Primer](https://learningtooptimize.github.io/LearningToControlClass/dev/class01/background_materials/math_basics.html) to help you brush up on essential concepts. This primer covers key topics such as matrix operations, eigenvalues, and eigenvectors besides other fundamental calculus concepts. It is recommended to review this primer before the first class.

### **Optimization**:
We will use JuMP for some optimization tasks. If you are new to JuMP, please review the [JuMP Tutorial](https://jump.dev/JuMP.jl/stable/tutorials/getting_started/getting_started_with_JuMP/) to familiarize yourself with its syntax and capabilities.

Test your knowledge with this (Pluto) [Modeling Exercise](https://learningtooptimize.github.io/LearningToControlClass/dev/class01/background_materials/optimization_basics.html).

*Final 🧠*: The (Pluto) [Motivational Exercise](https://learningtooptimize.github.io/LearningToControlClass/dev/class01/background_materials/optimization_motivation.html) will test what you have learned and motivate what we will research in the course.

## In-Class Material

Besides the administrative topics, we will cover the structure of the problem we are trying to solve and start with the basics of how to model it.
The main (Pluto) [Class 01 Notebook](https://learningtooptimize.github.io/LearningToControlClass/dev/class01/class01_intro.html) contains the in-class material.