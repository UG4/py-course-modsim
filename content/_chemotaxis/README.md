# Chemotaxis Model

## Governing Equations

This implementation solves a coupled system of reaction-diffusion equations on a 2D domain:

$$\frac{\partial u}{\partial t} = D_u \nabla^2 u$$

$$\frac{\partial v}{\partial t} = D_v \nabla^2 v - \gamma(a \cdot u - b \cdot v + w \cdot v^2)$$

$$\frac{\partial w}{\partial t} = D_w \nabla^2 w - \gamma(c \cdot u - v^2 \cdot w)$$

where:
- $u, v, w$ are concentration fields
- $D_u, D_v, D_w$ are diffusion coefficients
- $\gamma, a, b, c$ are reaction parameters
- $\nabla^2$ denotes the Laplacian operator

## Boundary Conditions

Dirichlet boundary conditions are applied on all domain boundaries:

$$u|_{\partial \Omega} = u_\infty$$

$$v|_{\partial \Omega} = v_\infty$$

$$w|_{\partial \Omega} = w_\infty$$

where $u_\infty, v_\infty, w_\infty$ are steady-state concentrations determined from equilibrium conditions.

## Initial Conditions

Random perturbations around steady-state values:

$$u(x, y, 0) = u_\infty + 0.1 \cdot [(e - c) \cdot \text{randint}(-1, 1) + c]$$

$$v(x, y, 0) = v_\infty + 0.1 \cdot [(e - c) \cdot \text{randint}(-1, 1) + c]$$

$$w(x, y, 0) = w_\infty + 0.1 \cdot [(e - c) \cdot \text{randint}(-1, 1) + c]$$

## Model Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| $D_u$ | 10.0 | Diffusion coefficient for $u$ |
| $D_v$ | 1.0 | Diffusion coefficient for $v$ |
| $D_w$ | 40.0 | Diffusion coefficient for $w$ |
| $a$ | 0.2 | Reaction parameter |
| $b$ | 1.0 | Reaction parameter |
| $c$ | 1.3 | Reaction parameter |
| $\gamma$ | 490.0 | Scaling parameter for reactions |
| $u_\infty$ | 1.0 | Steady-state concentration of $u$ |

## Domain

- Square 2D domain (`square2d.ugx`)
- Subsets: "Inner" (domain) and "Boundary" (boundary)
- Default refinement: 2 levels
