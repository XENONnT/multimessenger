## Scripts to automate simulation and processing

This directory contains several usefull scripts to simulate, 
process, analyse and plot with the multimessenger package.

Available scripts:
  * **simulate_snmodel.py** - simulates and/or fetches already 
simulated models. Run in the form:

<code bash>python simulate_snmodel.py -m {model_name} -i {model_index} -N {ntotal} -id {runid}</code>

  * **submit_sim.py** - submit a batch for simulating a given SN model. Run in the form:

<code bash>python submit_sim.py -m {model_name} -i {model_index} -N {ntotal} -id {runid}</code>