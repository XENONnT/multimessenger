
# SNAX (SuperNova Analysis in Xenon)

> Main responsible: Ricardo Peres (rperes@physik.uzh.ch), Melih Kara (kara@kit.edu), Amanda Depoian (adepoian@purdue.edu)
The go-to place when you doubt that DM exists but remember that both neutrinos and SN are definitely real. Welcome!
What to find:
  - Class to investigate Supernova Models
  - Class to create a target object (Xenon or Argon atoms)
  - Class to investigate interactions between a given model and a target

You can also use the SNAX package to investigate more things
  - Simulation code for SN in LXe experiments
  - From recoil spectrum to signal waveforms with wfsim.
  - Sensitivity and significance studies.

## Installation 

```
git clone git@github.com:XENONnT/multimessenger.git
cd multimessenger 
pip install ./
```

## Usage

snax model
```python
# get a model
from snax.SnewpyWrapper import SnewpyModel
from snax.models import SnaxModel
loader = SnewpyModel()
# get all possible combinations for a model
all_model_families = loader.display_models() ##"Nakazato_2013", "Bollig_2016" etc
parameter_combinations_df = loader.display_models("Nakazato_2013", verbose=False)
## load one specific model
nakazato_snax_id8 = loader.get_model("Nakazato_2013", combination_index=8)

# snax model with fluxes computed
sm = SnaxModel(nakazato_snax_id8)
sm.compute_model_fluxes()
```
Creating a target nuclei
```python
# create a target
from snax.Nucleus import Target
from snax.nuclei.Xenon_Atom import ATOM_TABLE

print("Target we load", ATOM_TABLE['Xe131'])
singleXe = Target(ATOM_TABLE['Xe131'], pure=True) # pure means setting the abundance to =1  
```
Creating interaction onjects
```python
# create interactions
from snax.interactions import Interactions
interaction = Interactions(sm, Nuclei="Xenon", isotope="mix") # uses the snax.Nucleus in the background 
interaction.compute_interaction_rates()
interaction.scale_rates(distance=10, volume=5.9)
interaction.plot_rates(scaled=True)

## and one can further create fuse instructions
from snax.Simulations import SimulateSignal
simulator = SimulateSignal(interaction, instruction_type="fuse_microphysics")
st, run_number = simulator.simulate_single()
# or get instructions and simulate manually
instructions_detphys = simulator.generate_fuse_detectorphysics_instructions()
instructions_micorphys = simulator.generate_fuse_microphysics_instructions()
```


Also see the [notebooks](./notebooks).

We are using [snewpy](https://github.com/SNEWS2/snewpy) for the supernova models.

Also check out the studies carried out in Supernova neutrino search context; <br>
[wiki index](https://xe1t-wiki.lngs.infn.it/doku.php?id=xenon:peres:sntrigger:snindex)<br>
[analysiscode](https://github.com/XENONnT/analysiscode/tree/master/Multimessenger)
