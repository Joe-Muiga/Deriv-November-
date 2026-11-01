"""rng_core - RNG-agnostic top-4 digit predictor (v2). Live code: numpy only, no imports from the bot.

Layers: observe (TickStore) -> diag (structure tests) -> exact (deterministic solvers) -> stat_model (stationary
Bayesian mixture + e-process) -> decide (top-4, calibration). orchestrator.py ties them per symbol; adapter.py is the
thin bridge to mh_engine. Nothing here knows about money, payouts or contract types.
"""
