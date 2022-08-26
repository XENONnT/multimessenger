
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
from multimessenger.supernova import Supernova_Models as sn
data = sn.Models(model_name='Fornax_2021')
```
Also see [notebooks](./notebooks).

By default, the composite is 'Xenon', later, Argon can also be implemented.

We are using [snewpy](https://github.com/SNEWS2/snewpy) for the supernova models, and 
they can simply be loaded by specifying the model name, and filename (or file index), if neither is given, the software 
displays your options and asks you to select one of them. Some models might be taking key-word arguments, which can be passed
by `model_kwargs` argument to the `sn.Models()` function.


For more see [notebooks](./notebooks).
