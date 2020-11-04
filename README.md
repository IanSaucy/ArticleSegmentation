# CS501-Liberator-Project
Project Code for CS501 Libeartor Project

# Model
Credit for the original base: https://github.com/poke1024/bbz-segment
We hae some outstanding upstream commits that we will open a PR for soon. 

### SCC Usage \<TODO>
If you want to skip the setup of the below you can run directly on the SCC. 

You need to load `python3/3.7.7` and put the shared python packaged library on your python path. 

## Setup

- Python 3.7
- Pipenv
- Pre-trained model from dropbox [here](https://www.dropbox.com/sh/4b1ub2bmmgmbprp/AAC88d8h8oZVgt-4WC5_uNloa?dl=0)

Download and extract the model to `05_prediction/data/models`

Navigate to `bbz-segment` and run `pipenv install`

## Testing

Run inside a pipenv shell:

> `cd 05_prediction`

> `python src/main.py data/pages data/models`

Segmented images saved to `./data` root.