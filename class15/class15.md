# Class 15 — 10/03/2025

**Presenter:** Shuaicheng (Allen) Tong

**Topic:** Dynamic Optimal Control of Power Systems; Generators swing equations, Transmission lines electromagnetic transients, dynamic load models, and inverters.

---

# Overview

This chapter introduces the foundation of electric power system dynamic behaviors and shows how they are incorporated into modern optimal control formulations such as Transient Stability–Constrained Optimal Power Flow (TSC-OPF). We begin with the physics of electromagnetic transients to motivate the formulation of TSC-OPF, move through generator and inverter dynamics, and conclude with dynamic load models that capture how demand responds during disturbances. Together, these components are key to understand, simulate, and optimize real-world power system.

# Materials

The chapter is implemented as a [Pluto notebook](https://learningtooptimize.github.io/LearningToControlClass/dev/class15/class15.html), which contains derivations, visuals, and algorhtmic examples. To use the interactive plot, please run the notebook locally. Refer to [Class 01 documentation](https://learningtooptimize.github.io/LearningToControlClass/dev/class01/class01/#Background-Material). In Step 3, run `Pkg.activate()` with the path of class15 folder. The [lecture slide](https://learningtooptimize.github.io/LearningToControlClass/dev/class15/class15_lecture.pdf) contains the material of the video recording.

# Topics Covered

## Transients and Electromagnetic Dynamics
- Physical origin of transients in power system 
- Connection between Faraday's law, inductors/capacitors, and transient behavior  
- Relation of time-domain differential equations to steady-state phasor models  
- Introduction to transmission-line dynamics and telegrapher’s equations  

## Generator Swing Equation
- Rotor acceleration and deceleration under power imbalance  
- Role of inertia in stabilizing frequency  
- Per-unit formulation and damping effects  

## Inverter Dynamics and Grid Control
- Differences between synchronous generators and renewable inverters  
- Grid-following vs. grid-forming behavior  
- Virtual inertia and frequency droop control for renewable integration  

## Dynamic Load Models
- Induction motor dynamics: slip, torque imbalance, and stalling behavior  
- Voltage recovery models such as Exponential Recovery Load (ERL)  
- Differences between physics-based motor models and empirical aggregate models  

## Transient Stability–Constrained Optimal Power Flow (TSC-OPF)
- Time-domain constraints ensuring system stability during disturbances  
- Solution methods such as direct transcription and multiple-shooting formulations  
- Forward and adjoint sensitivity analysis for efficient gradient calculation

# Learning Objectives

By the end of this chapter, readers will be able to:
- Describe the physical origins of transients and how they propagate in power networks  
- Explain generator swing dynamics and how they regulate grid frequency  
- Understand how inverter controls emulate generator behavior
- Distinguish between static and dynamic load models and when each is appropriate
- Understand the role of sensitivity analysis in dynamic optimal control

