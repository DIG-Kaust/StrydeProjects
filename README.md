![LOGO](https://github.com/DIG-Kaust/StrydeProjects/blob/main/logo.png)

This repository contains all the routines that our group as created to manipulate and visualize SEG-Y data produced by STRYDE SeismicQC software.
Moreover, it contains all the notebooks created to perform basic analysis of the data acquired over time.

## Project structure
This repository is organized as follows:

* :open_file_folder: **pystride**: python library containing basic routines for data manipulation and visualization;
* :open_file_folder: **data**: folder containing links to the datasets acquired by our group (and used in the notebooks);
* :open_file_folder: **notebooks**: set of jupyter notebooks performing basic analysis of the different datasets;


## Getting started :space_invader: :robot:
To ensure reproducibility of the results, we suggest using the `environment.yml` file when creating an environment.

Simply run:
```
./install_env.sh
```
It will take some time, if at the end you see the word `Done!` on your terminal you are ready to go. After that you can simply install your package:
```
pip install .
```
or in developer mode:
```
pip install -e .
```

Remember to always activate the environment by typing:
```
conda activate strydeenv
```
