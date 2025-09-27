# Contextual Failure discovery using Bayesian Experiment Design
Repository for "Cost-aware Discovery of Contextual Failures using Bayesian Active Learning", CoRL 2025

# CARLA failure discovery for Mode 1 (misdetection due to bad visibility from distance) and Mode 2 (misdetection due to bad lighting)
This example uses a GiT+LLM for failure diagnosis as an expert
Instructions:
1. Clone the repo
2. Create conda environment and execute the following script to check if things are working: 
``` 
    conda env create -f environment.yml
    conda activate cfail
    python3 scripts/carla/run_eci.py --seed SEED --num_init N1 --num_iter N2 --delta_light D1 --delta_dist D2 --radius R
```
Use `SEED`, `N1`,`N2` to control the seed, number of iterations for initializing the prior using random sampling, number of iterations for Bayesian loop. 
Control the severity of each failure mode using `D1` and `D2`, (0 means low threshold on severity, 1 means high). We estimate severity in this example using the number of images in a simulation where misdetection happens. Min value should be `0.1`, which corresponds to atleast one image with failure per simulation. `R` controls the sampling resolution, aka neighbourhood of each scenario, setting `R` higher will lead to broader sampling, low `R` will lead to finer sampling.
