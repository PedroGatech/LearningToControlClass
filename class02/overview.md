# Class 2 — 08/29/2025

**Presenter:** Arnaud Deza

**Topic:** Numerical optimization for control (gradient/SQP/QP); ALM vs. interior-point vs. penalty methods

---

## Overview

This class covers the fundamental numerical optimization techniques essential for optimal control problems. We explore gradient-based methods, Sequential Quadratic Programming (SQP), and various approaches to handling constraints including Augmented Lagrangian Methods (ALM), interior-point methods, and penalty methods.


The slides for this lecture can be found here [Lecture Slides (PDF)](https://learningtooptimize.github.io/LearningToControlClass/dev/class02/ISYE_8803___Lecture_2___Slides.pdf)

The Pluto julia notebook for my final chapter can be found here [final chapter](https://learningtooptimize.github.io/LearningToControlClass/dev/class02/class02.html) 

Although the main code for the julia demo's are contained in the Pluto notebook above, the following julia notebooks are the demo's I used in the class recording/presentation.

 
1. **[Part 1a: Root Finding & Backward Euler](https://learningtooptimize.github.io/LearningToControlClass/dev/class02/part1_root_finding.html)**
   - Root-finding algorithms for implicit integration
   - Fixed-point iteration vs. Newton's method
   - Backward Euler implementation for ODEs
   - Convergence analysis and comparison
   - Application to pendulum dynamics

2. **[Part 1b: Minimization via Newton's Method](https://learningtooptimize.github.io/LearningToControlClass/dev/class02/part1_minimization.html)**
   - Unconstrained optimization fundamentals
   - Newton's method for minimization
   - Hessian matrix and positive definiteness
   - Regularization and line search techniques 

3. **[Part 2: Equality Constraints](https://learningtooptimize.github.io/LearningToControlClass/dev/class02/part2_eq_constraints.html)**
   - Lagrange multiplier theory
   - KKT conditions for equality constraints
   - Quadratic programming with equality constraints 

4. **[Part 3: Interior-Point Methods](https://learningtooptimize.github.io/LearningToControlClass/dev/class02/part3_ipm.html)**
   - Inequality constraint handling
   - Barrier methods and log-barrier functions
   - Interior-point algorithm implementation 
  
---

*For questions or clarifications, please reach out to Arnaud Deza at adeza3@gatech.edu*
