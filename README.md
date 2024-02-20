# ICPE 2024 Data Challenge Replication Package

This repository contains the replication package for the ICPE 2024 Data Challenge paper titled _Grammar-Based Anomaly Detection of Microservice Systems Execution Traces_

## Structure

The repository is structured as follows:

- `data/`: contains the datasets provided by Traini et al. in <https://github.com/SpencerLabAQ/icpe-data-challenge-delag>
- `rq1/`: contains the scripts and results for the first research question
- `rq2/`: contains the scripts and results for the second research question

## Setup

To replicate the results, you need to have [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed. Then, you can create a new environment with the required dependencies using the following command:

```bash
conda env create -f environment.yml
```

## Replication

Refer to the Jupyter Notebooks in the `rq1/` and `rq2/` folders to replicate the results of the paper.
