# Team Topological Ducks
Members: Jeff Burka, Atharv Chowdhary, Hugo Mullen, Simon Nirenberg. This is our submission for IQM's challenge at iQuHACK 2026.

## Repo Structure && Important Files
The most important files to look out for can be found in the `working/` directory. Here you can see the two most important witnesses we used in `linear_witness.py` and `ghz_witness.py`. You can also take a look at some of the hardware optimization techniques we employed in `kruskal.py`, `dijkstra.py`, and `optimize.py`. `simulate.py` allows us to easily integrate the optimization methods with our witness implementations on quantum hardware. `random_graph_witness.py` is a crucial script that allows the user to generate arbitrary graph states, which we used as a means of data generation for our benchmarking task.

You can also check out some of the other witnesses we tried to use in the `witnesses/` directory, as well as some variants of the witness algorithms we tried to implement. `calibration_data/` simply contains data from IQM's website and scripts for parsing data, and `visualization/` contains scripts for visualizing some of the data we gathered.

## Background
We use entanglement witnesses for two primary classes of quantum circuits, as outlined in [this]([https://arxiv.org/pdf/quant-ph/0405165]) paper.

## Approach
We approached this challenge with two primary goals in mind:
1. Devise and implement (in code) a task that can act as a benchmark for multipartite entanglement capabilities on quantum hardware.
2. Push IQM's own quantum hardware as far as possible on the benchmark we create.

For the first goal, we used `random_graph_witness.py` 

## Results

## Optimization Techniques
