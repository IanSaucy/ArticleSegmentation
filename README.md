# Grader Note
We moved the code directly relating to the model into a sub-module -- which means that changes are not directly reflected here in this repo until we pull those update into one of the branches. As such, you should also check that module directly(it's a public repo).

If there are any question ask Ian since he(me) set that bit up!

# CS501-Liberator-Project
Project Code for CS501 Libeartor Project

# Model
Credit for the original base: https://github.com/poke1024/bbz-segment
We have forked the project and are implementing our own changes. This fork is setup as a sub-module under Git.

### SCC Usage \<TODO>
If you want to skip the setup of the below you can run directly on the SCC. 

You need to load `python3/3.7.7` and put the shared python packaged library on your python path. 

## Setup

- Python 3.7
- Pipenv
- Pre-trained model from dropbox [here](https://www.dropbox.com/sh/7tph1tzscw3cb8r/AAA9WxhqoKJu9jLfVU5GqgkFa?dl=0)
- OCR model if using tesseract. Install [Google Tesseract OCR](https://github.com/tesseract-ocr/tesseract) (additional info how to install the engine on Linux, Mac OSX and Windows).
- config.json file with ABBYY API ID, Password, and server url if you are using ABBYY OCR! [ABBYY OCR](https://www.ocrsdk.com/), also must have the code to run the ABBYY Online SDK downloaded and include the path to that and that config.json file when using it

Download and extract the model to `05_prediction/data/models`

Navigate to `bbz-segment` and run `pipenv install`

## Testing

Run inside a pipenv shell:

> `cd 05_prediction`

> `python src/main.py data/pages data/models`

Segmented images saved to `./data` root.
