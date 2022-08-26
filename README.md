# FraudDetection_MLH

## Setting up an environment

**Pip** <br/>
Use ```pip install -r /path/to/requirements.txt``` to install the packages to run the model.

**Conda** <br/>
Conda uses an environment.yaml file instead of requirements.txt, but you can include one in the other:

##### environment.yaml
name: test-env
dependencies:
  - python>=3.5
  - anaconda
  - pip
  - pip:
    - -r file:requirements.txt

Then use conda to create the environment via

```conda env create -f environment.yaml```

## Executing

Run the main.py file to train the model and return test scores.
