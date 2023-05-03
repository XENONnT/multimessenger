
# multimessenger

> Main responsibles: Ricado Peres (rperes@physik.uzh.ch), Melih Kara (kara@kit.edu), Amanda Depoian (adepoian@purdue.edu)

The go-to place when you doubt that DM exist but remember that both neutrinos and SN are definetly real. Welcome!
What to find:
  - Simulation code for SN in LXe experiments
  - From recoil spectrum to signal waveforms with wfsim.
  - Sensitivity and significance studies

## Installation 

```
git clone git@github.com:XENONnT/multimessenger.git
cd multimessenger 
pip install ./
```

## Usage

```python
# get a model
from snax import Supernova_Models

SN_Nakazato = Supernova_Models.Models("Nakazato_2013", config_file="./local_conf.conf")
SN_Nakazato(index=5)  # load a progenitor (brings the attributes)
SN_Nakazato.compute_model_fluxes()  # calculate the fluxes for a set of param
fluxes_at10 = SN_Nakazato.scale_fluxes(distance=10)  # scale fluxes
```

```python
# create a target
from multimessenger.supernova.Nucleus import Target
from snax.Xenon_Atom import ATOM_TABLE

singleXe = Target(ATOM_TABLE['Xe131'], pure=True)  # pure means setting the abundance to =1 
```
```python
# create interactions
from multimessenger.supernova.interactions import Interactions
Int = Interactions(SN_Nakazato, Nuclei='Xenon', isotope='Xe131') # isotop=string creates a TARGET
Int.compute_interaction_rates()
Int.plot_rates(scaled=False)
```


Also see the [notebooks](./notebooks).

By default, the composite is 'Xenon', later, Argon can also be implemented.

We are using [snewpy](https://github.com/SNEWS2/snewpy) for the supernova models, and 
they can simply be loaded by specifying the model name, and filename (or file index), if neither is given, the software 
displays your options and asks you to select one of them. Some models might be taking key-word arguments, which can be passed
by `model_kwargs` argument to the `sn.Models()` function.

Also check out the studies carried out in Supernova neutrino search context; <br>
[wiki index](https://xe1t-wiki.lngs.infn.it/doku.php?id=xenon:peres:sntrigger:snindex)<br>
[analysiscode](https://github.com/XENONnT/analysiscode/tree/master/Multimessenger)
