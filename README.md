# Orbital Propagator

A Python-based orbital mechanics simulation tool that implements multiple numerical integration methods to propagate satellite orbits and compare their accuracy against reference data.

## Overview

This project provides a complete framework for orbital propagation using three different numerical integration methods:
- **Euler Method**: First-order integration (simple but less accurate)
- **Verlet Method**: Second-order symplectic integration (good energy conservation)
- **Runge-Kutta 4th Order (RK4)**: Fourth-order integration (high accuracy)

The propagator supports both dimensionalized and non-dimensionalized formulations of the orbital equations of motion, allowing for numerical stability analysis and performance comparison.