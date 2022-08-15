Dfisher_2022A Documentation
=============

This project is being developed in the course of delivering the DFisher_2022A ADACS Merit Allocation Program project.

## Installation

#### Pre-requirement:
* python >=3.8 <3.10
* HDF5 >= 1.8.4 (>=1.8.15 is strongly recommended)

#### Latest PyPI release 
```
$ pip install dfisher_2022a
```
**Common troubleshooting**: If installation fails, try to upgrade `pip` by running `pip install --upgrade pip` first.
#### Latest dev-version on GitHub
```
$ pip git+https://github.com/ADACS-Australia/dfisher_2022a.git#egg=dfisher_2022a
```
**NOTICE**: In the dev-version, a faster version of `lmfit` ([light-lmfit-py](https://github.com/ADACS-Australia/light-lmfit-py/tree/light)) is used. This version provides a fitting method, "fast_leastsq" in addition to other [fitting methods](https://lmfit.github.io/lmfit-py/fitting.html#choosing-different-fitting-methods) available in `lmfit(1.0.3)`. This method can be 2x faster than `leastsq`. Check [dev notes](https://github.com/ADACS-Australia/light-lmfit-py/tree/light) for more details.

## Getting Started
##### Import the package
```
>>> import dfisher_2022a
```
#### Read in data cube
```
>>> cube = dfisher_2022a.ReadCubeFile("single_gaussian_muse_size.fits").cube
```
If a separate variance file is provide:
```
>>> cube = dfisher_2022a.ReadCubeFile("single_gaussian_muse_size.fits", "muse_var.fits").cube
```
#### Prepare data for fitting
```
>>> p = dfisher_2022a.ProcessedCube(cube, z=0.009, snr_threshold=5.)
```
##### 1. De-redshift the cube
```
>>> p.de_redshift()
```
##### 2. Select fitting region for a given line
```
>>> p.select_region("Halpha", left=20, right=20)
```
Keywords `left` and `right` set the wavelength cuts around the given line on both sides, e.g. the selected region is [line-left, line+right]. If this region exceeds the cube wavelength range, a nearest value within the cube will be used instead.

##### 3. Filter the cube by SNR threshold
```
>>> p.get_snrmap()
```
#### Select fitting model
```
>>> model = dfisher_2022a.Lm_Const_1GaussModel
```
A single Gaussian model is available within this package.

#### Fit the cube
```
>>> cfl = dfisher_2022a.CubeFitterLM(data=p.data, weight=p.weight, x=p.x, model=model, method='leastsq') # accept lmfit.Model.fit kwargs
>>> cfl.fit_cube()
```
Additional keyword arguments for [lmfit.Model.fit](https://lmfit.github.io/lmfit-py/model.html#model-class-methods) can be passed to the class object as well.

#### Save output
```
>>> out = dfisher_2022a.ResultLM()
>>> out.get_output(p) # get attributes from ProcessedCube object
>>> out.get_output(cfl)
>>> out.save()
```
An `out` directory will be generated in the current directory.

#### Read output
In the `.out` folder:
```
result.h5
fitdata/
```
where `result.h5` stores the fitting result, and `fitdata/` contains processed data used for fitting.

   To read `result.h5` file:
   ```
   >>> import pandas as pd
   >>> store = pd.HDFStore("result.h5")
   >>> store.keys()
   ['/Halpha_Const_1GaussModel']
   >>> df = store.get("Halpha_Const_1GaussModel")
   ```

#### Check available lines
```
>>> dfisher_2022a.EmissionLines
{'Halpha': 6562.819, 'Hb4861': 4861.333, 'Hdelta': 4101.742, ...
```
The line information is included in `emission_lines.py`. Users can customize this file (e.g. adding more lines or updating the wavelength) before importing this package.

#### A wrapped approach

A wrapper function encapsulating steps 1-6 is available:
```
>>> from dfisher_2022a import fit_lm
>>> model = dfisher_2022a.Lm_Const_1GaussModel
>>> fit_lm(cubefile="single_gaussian_muse_size.fits", line="Halpha", model=model, z=0.009, left=20, right=20, snr_threshold=5.)
```
#### Use the faster version of lmfit
If dev-version of this package is installed, which uses a faster version of `lmfit` as dependency, a faster fitting method is also available, by using `method="fast_leastsq"`and adding an argument `fast=True`
```
>>> cfl = dfisher_2022a.CubeFitterLM(data=p.data, weight=p.weight, x=p.x, model=model, method='fast_leastsq', fast=True) # accept lmfit.Model.fit kwargs
>>> cfl.fit_cube()
```
In the wrapper function:
```
>>> fit_lm(cubefile="single_gaussian_muse_size.fits", line="Halpha", model=model, z=0.009, left=20, right=20, snr_threshold=5., method="fast_leastsq", fast=True)
```

## Create custom model
Users can create their own models following the descriptions provided by [lmfit](https://lmfit.github.io/lmfit-py/model.html). To use `fast_leastsq` method in the dev version, `eval_fast` needs to be written as a method of the model. See dev notes of [light-lmfit-py](https://github.com/ADACS-Australia/light-lmfit-py/tree/light)  for more details.