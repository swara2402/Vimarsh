# Milestone 59: Constrained verification-action allocation

## Purpose
Allocate a batch of candidate verification actions using positive conservative benefit diagnostics, while enforcing one action per case, a shared budget, and optional per-action capacity limits.

## Optimization
The binary optimization selects candidate-action pairs to maximize the sum of supplied lower-bound benefit diagnostics subject to:
- at most one action per case;
- total selected cost no greater than the shared budget;
- per-action capacity limits;
- only actions marked available with strictly positive benefit diagnostics are eligible.

The implementation uses `scipy.optimize.milp`. If the solver does not return a proven successful solution, the safe fallback is no allocation.

## Important boundaries
This is a batch allocator for offline research and policy comparison. It does not establish that benefit estimates are calibrated, causal, or safe. It must only receive benefit diagnostics produced from valid, completed action-outcome evidence. The optimizer does not execute actions, create providers, or authorize unsafe actions. The separate safety/availability layer remains authoritative.
