import os

# Sampling runs on JAX (numpyro); pytensor's C backend is only used for graph
# rewrites and fails to compile with recent macOS clang, so it is turned off.
os.environ.setdefault("PYTENSOR_FLAGS", "cxx=")
