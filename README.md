# Posterior Predictive Null Checks (PPN)

This repository contains the code for the paper "Posterior Predictive Null Checks" (Moran et al. 2021).

## Requirements

The R code has been tested on R version 4.0.2 with the following packages:
```bash
MASS
tidyverse
ggtext
ggpubr
philentropy
LaplacesDemon
ggplot2
dplyr
reshape2
grid
gridExtra
gtable
xtable
RColorBrewer
kSamples
reticulate
```

The Python code has been tested on Python 3.7.6 with the packages listed in `factor-analysis/environment.yml`.


## Index

- `gaussian-mixture`

This folder contains code to replicate the Gaussian mixture model example in Section 1. The main file is `gaussian-mixture/analysis.R`.

- `regression`

This folder contains code to replicate the regression example in Section 2.5.

- `multinomial-mixture`

This folder contains code to replicate the multinomial mixture model experiment in Section 3.1. The main file is `multinomial-mixture/analysis.R`. The data can be found in `multinomial-mixture/dat`. The data was sourced from Table 1 of Stern et al. 1995.

- `factor-analysis`

This folder contains code to replicate the factor analysis examples in Sections 3.2 and 3.3. The code for Section 3.2 is `linear_example.py` with additional plotting of results in `plot_linear_example.R` 

The code for Section 3.2 is `nonlinear_example.py` with additional plotting of results in `plot_nonlinear_example.R` 


## References
- Moran, G.E., Cunningham, J.P., and Blei, D. M.  (2021) Posterior Predictive Null Checks. arXiv.
- Stern, H. S., Arcus, D., Kagan, J., Rubin, D. B., and Snidman, N. (1995). Using mixture models in temperament research. International Journal of Behavioral Development, 18(3): 407–423. 13, 14.


