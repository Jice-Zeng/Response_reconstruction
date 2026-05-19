This repository provides the implementation of a diffusion-model-based framework for probabilistic dynamic response reconstruction. The proposed method reconstructs full structural responses from sparse, partial, or missing measurements and provides posterior uncertainty estimates for the reconstructed responses.

Structural health monitoring (SHM) systems often suffer from incomplete measurements due to sparse sensor deployment, sensor malfunction, communication loss, or missing data. This repository implements a diffusion-based posterior reconstruction approach that uses a learned generative prior to reconstruct the full dynamic response field from limited observations.

The framework supports different observation settings, including:

- Fixed-step subsampling
- Block-wise missing data
- Random subsampling
- Sparse sensor layouts
- Noisy measurements
