# Notebooks for Supernova Analysis

The scripts are under `xenonnt/multimessenger/supernova/`. 

[//]: # ()
[//]: # (The notebooks;)

[//]: # (- [SN Signal - 1]&#40;Supernova_Signal-1.ipynb&#41; <br>)

[//]: # (Demonstrates how to load existing simulated SN data. These data have neutrino fluxes and luminosities at different )

[//]: # (time steps and from several energy channels for three neutrino flavors. )

[//]: # (The `multimessenger.supernova` package provide the tools to load and visualize this data. Furthermore, the interactions )

[//]: # ( with the Xenon nuclide are included.)

[//]: # (After each recoil calculation &#40;`get_recoil_spectra1D&#40;&#41; / get_recoil_spectra2D&#40;&#41;`&#41; it saves the results in the object)

[//]: # ( such that it can  be fetched next time it runs. <br> This is particularly useful for 2D data where it computes )

[//]: # (the rates at 300+ time steps and for several recoil energies, and can take up to hours.)

[//]: # (- [SN Signal - 2]&#40;Supernova_Signal-2.ipynb&#41; <br>)

[//]: # (Demonstrates the 2 dimensional interaction rates. )

[//]: # (`get_recoil_spectra1D&#40;&#41;` computes the rates for a progenitor at a given distance for a given nuclide, )

[//]: # (integrated over the specified time interval e.g. 0-10 sec. )

[//]: # (<br> Whereas `get_recoil_spectra2D&#40;&#41;` computes the rates at each individual time steps giving a time distribution)

[//]: # (as well. )

[//]: # (These rates, can later be integrated over a given time interval to get the same results &#40;see `_get_1Drates_from2D&#40;&#41;`&#41; <br>)

[//]: # (The integrated and individual rates are also visualized. 3 dimensional plots showing the rates for each flavor at each recoil energy and at each time can be found in this notebook.)

[//]: # (- [SN Signal - 3]&#40;Supernova_Signal-3.ipynb&#41; <br>)

[//]: # (Reads in the computed data and demonstrates sampling from the recoil energies and times. Shows how the object attributes can easily be imported and passed to other functions.)

[//]: # (> There is also a `Simulate_Signal.Simulator` which provides a basic light and charge yields and using nest corrected S1 and S2 areas. )

[//]: # (> However, for more realistic XENONnT simulations refer to next notebooks that use [WFSim]&#40;https://github.com/XENONnT/WFSim&#41;.)

[//]: # ()
[//]: # (- [WFSim instructions]&#40;.&#41;)

[//]: # ()
[//]: # ()
[//]: # ()
[//]: # (> Notes: <br>)

[//]: # (> The Xenon composite that is used for the recoil calculation can be found in [here]&#40;../multimessenger/supernova/Xenon_Atom.py&#41;)