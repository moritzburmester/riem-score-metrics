# Score-Based Riemannian Metrics 
GitHub Repository for my Master Thesis on Score-Based Riemannian Metrics from Diffusion Models

## Overview 
This repository contains the code necessary to reproduce the experiments from my thesis. Specifically, I investigated the behavior of different Riemannian metrics from literature, as well as proposed a new metric that interpolates between manifold-aware and density-aware terms. The manifold-aware term is composed of the Jacobian of the score function, where as the density-aware term is the magnitude of the score function. The Jacobian term guides geodesics to move tangentially to the underlying data manifold, where as the magnitude term ideally pulls the geodesic towards higher density regions.

## Repository Structure 
```
├── assets 
├── diffusion_model_dependencies/   # scripts and modules to train, evaluate, and sample from score-based diffusion models
├── experiments_toy_datasets/       # geodesics under different metrics for the toy datasets (circle, s-curve, swiss-roll/spiral, ucg, wcg, two moons)
├── experiments_urc/                # geodesics on the uniform rotated characters (URC) dataset
├── experiments_morphbench/         # geodesics in the stable diffusion latent space
├── tutorial.ipynb                  # notebook to optimize geodesics under different Riemannian metrics on the toy datasets
├── requirements.txt
└── README.md
```
## Example Geodesics

### URC (Uniform Rotated Characters)

<table>
  <tr>
    <td align="right" width="120"><b>LERP</b></td>
    <td><img src="assets/LERP_g18.png" width="850"></td>
  </tr>
  <tr>
    <td align="right"><b>SLERP</b></td>
    <td><img src="assets/SLERP_g18.png" width="850"></td>
  </tr>
  <tr>
    <td align="right"><b>λ = 0</b></td>
    <td><img src="assets/lam0_g18.png" width="850"></td>
  </tr>
  <tr>
    <td align="right"><b>λ = 0.1</b></td>
    <td><img src="assets/lam0.1_g18.png" width="850"></td>
  </tr>
  <tr>
    <td align="right"><b>λ = 1</b></td>
    <td><img src="assets/lam1_g18.png" width="850"></td>
  </tr>
  <tr>
    <td align="right"><b>LAND</b></td>
    <td><img src="assets/LAND_g18.png" width="850"></td>
  </tr>
  <tr>
    <td align="right"><b>RBF</b></td>
    <td><img src="assets/RBF_g18.png" width="850"></td>
  </tr>
</table>

### MorphBench (M) in Stable Diffusion v2.1-base 

<table>
  <tr>
    <td align="right" width="120"><b>LERP</b></td>
    <td><img src="assets/mbm_wolf_tiger_lerp_strip.png" width="850"></td>
  </tr>
  <tr>
    <td align="right"><b>SLERP</b></td>
    <td><img src="assets/mbm_wolf_tiger_slerp_strip.png" width="850"></td>
  </tr>
  <tr>
    <td align="right"><b>λ = 0</b></td>
    <td><img src="assets/mbm_wolf_tiger_lam00_it500_strip.png" width="850"></td>
  </tr>
  <tr>
    <td align="right"><b>λ = 0.1</b></td>
    <td><img src="assets/mbm_wolf_tiger_lam01_it500_strip.png" width="850"></td>
  </tr>
  <tr>
    <td align="right"><b>λ = 0.25</b></td>
    <td><img src="assets/mbm_wolf_tiger_lam025_it500_strip.png" width="850"></td>
  </tr>
  <tr>
    <td align="right"><b>λ = 0.5</b></td>
    <td><img src="assets/mbm_wolf_tiger_lam05_it500_strip.png" width="850"></td>
  </tr>
  <tr>
    <td align="right"><b>λ = 0.75</b></td>
    <td><img src="assets/mbm_wolf_tiger_lam075_it500_strip.png" width="850"></td>
  </tr>
  <tr>
    <td align="right"><b>λ = 1</b></td>
    <td><img src="assets/mbm_wolf_tiger_lam10_it500_strip.png" width="850"></td>
  </tr>
</table>


