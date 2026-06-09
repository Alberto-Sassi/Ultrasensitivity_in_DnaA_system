# Ultrasensitivity_in_DnaA_system

These codes were used to obtain the results shown in "Sassi and Pigolotti, Ultrasensitive response in bacterial replication initiation, bioRxiv 2026".

The parameters for the simulations are contained in files such as "src/configs/base_tau=25.yaml".
For a single run, use (in case for example tau=25 min):

$ python -m experiments.run_single --config src/configs/base_tau=25.yaml

For a parameter sweep, with several values of the cooperativity and the dissociation constant, the choice of the range of parameters is possible in "src/configs/sweeps/cooperativity_binding_energy.yaml"

To run the parameter sweep, use (in case for example tau=25 min)

$ python -m experiments.sweeps.run_sweep --config src/configs/base_tau=25.yaml --sweep src/configs/sweeps/cooperativity_binding_energy.yaml

The time traces are saved in a .jason file in the "results" directory.  
