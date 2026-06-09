# Ultrasensitivity_in_DnaA_system

This repository contains the code used to generate the results presented in "Sassi and Pigolotti, Ultrasensitive response in bacterial replication initiation, bioRxiv 2026".

Simulation parameters are specified in configuration files located in src/configs/, for example src/configs/base_tau=25.yaml.
To run a single simulation, execute the following command from the root directory of the repository (in case for example tau=25 min):

$ python -m experiments.run_single --config src/configs/base_tau=25.yaml

Parameter sweeps -several values of cooperativity and dissociation constant K_ori- can be configured in
"src/configs/sweeps/cooperativity_binding_energy.yaml"

To run a parameter sweep, use (i.e. for tau=25 min)

$ python -m experiments.sweeps.run_sweep --config src/configs/base_tau=25.yaml --sweep src/configs/sweeps/cooperativity_binding_energy.yaml

Simulation outputs (time traces) are saved in JSON format in the results/ directory.
