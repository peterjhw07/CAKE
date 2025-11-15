# CAKE

Continuous Alteration Kinetic Elucidation (CAKE) is an experimental technique for analyzing the kinetics of reactions performed under continuous altering conditions, such as reagent concentration or temperatue.

Pythonic code for processing experimental CAKE data can be downloaded for offline use and modification here. The code is also implemented in a web application: http://www.catacycle.com/cake.

The technique and its application are published at https://doi.org/10.1039/D3SC02698A and pre-printed at DOI: 10.26434/chemrxiv-2022-52g7z. Please cite this publication should the technique and this code be used in your work.

## Installation

The code can be downloaded as a package for offline use using the conventional github download tools.

## Usage

Various examples uses of cake are featured under "examples". Tutorial videos on usage are a work in progress.

Use help(function) for documentation for the principle functions below.

### Principle functions

```python
# Import package
import cake

# Simultate experimental CAKE data
cake.sim(**kwargs)

# Fit expeirmental CAKE data
cake.fit(**kwargs)
```
