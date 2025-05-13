# cost of living prediction exercise by area

## How to use
Create a virtual environment using:
```
python -m venv .venv
```
Install the project dependencies:
```
pip install -r requirements.txt
```

## Required files
It's required to download the data files from kaggle Data Science competition
* mobility_data.parquet
* train.csv
* test.csv

And place that files in ``./data`` directory.

## Running test
Use ``python e10.py`` if your hardware can handle 36GB of RAM usage
Use ``python e10-chunked.py`` for 4.2GB of RAM usage

In ``e10-chunked.py`` file you can adjust ``BATCH_SIZE`` to lower or rise the usage of RAM 18,000,000 for upto 4.2GB available RAM

## What is in utils folder?
There is some sample code to perform basic and reference operations
* ``combine.py`` combine multiple parquet files
* ``h3-test.py`` to test the H3 functionality
* ``re-training`` for a basic model 2stages training
* ``training-best-params.py`` for test some training params, find the best fit for current hardware