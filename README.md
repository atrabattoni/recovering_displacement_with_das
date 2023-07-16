# Recovering displacement from distributed acoustic sensing data for seismic applications

This repository contains the codes to reproduce the figures presented in the following article:

> Trabattoni, A., Biagioli, F., Strumia, C., van den Ende, M., Scotto di Uccio. F., Festa, G., Rivet, D., Sladen, A., Ampuero J. P., Métaxian, J. P., Stutzmann, E. (2023). **Recovering displacement from distributed acoustic sensing data for seismic applications**. *Geophysical Journal International*.

To run the codes first install and activate the conda environment with:

```
conda env create -f environment.yml
conda activate trabattoni2023
```

Then run all the scripts by simply executing:

```
sh run.sh
```

Data are downloaded from a Zenodo repository. Notes that the codes relies on [**xdas**](https://github.com/xdas-dev/xdas), a python library for DAS data processing.