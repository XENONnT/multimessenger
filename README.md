
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
import multimessenger.supernova as sn 
sn_model  = sn.Supernova_Models.SN_LightCurve(composite='Xenon', 
                                              config_file='../multimessenger/simple_config.conf')
sn_model.load_model_from_db(progenitor_mass = 30,
                            metallicity= 0.02,
                            time_of_revival = 100,)
```
Also see [multimessenger/notebooks](multimessenger/notebooks).

By default, the composite is 'Xenon', later, Argon can also be implemented. If running on dali, `SN_LightCurve` can 
be called with default arguments. In which case, it fetches the basic configuration file from `dali/lgrandi/melih/mma/data/basic_conf.conf`

We are using [this SN database](http://asphwww.ph.noda.tus.ac.jp/snn/) for the supernova models, and 
they can simply be loaded by specifying the `progenitor_mass, metallicity`, and `time_of_revival`
 when calling `load_model_from_db` function. Later, other models (such as those from [snewpy](https://github.com/SNEWS2/snewpy) 
can be integrated. 

There is a _very basic_ wfsim simulation, and investigation method available. 

For more see [multimessenger/notebooks](multimessenger/notebooks).
