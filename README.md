# README
(c) 2016

Author: Aidan Ross

Email: aidanrwross@gmail.com

## Analysis for IHC stained images

1. Python Package Dependencies
------------------------------

- Numpy
- Matplotlib
- SciKit Image
- Scipy
- Glob

2. Structure
------------

a) subfoler importable as Python module

b) tools subfolder contains main pipeline

c) user can specify paths and subjective paramters in paramters.py script

d) main calling done in run.py script

e) test/sample data

3. Running main script
----------------------

Four main methods called:
* Displaying Images
generally only used for testing and accuracy checks
* Saving Images
* Get Data
* Write Data to CSV
  - write CSV with pixel data
  - write CSV with micron data
    - In parameters file script, specify the resolution of the Images being analyzed/ scanner used to obtain TMA images

4. Disclaimer
-------------

These scripts were developed for a particular data set, all functions may not be error free
