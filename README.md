# Newspaper Article Segmentation

This repository contains tools for segmenting archives of newspapers, in the form of digital images into individual articles containing various article level information, such as content, author, title etc. It utilizes a DNN and a set of post-processing scripts to segment images into articles, producing a JSON output that contains the detected information. 

Currently, it supports segmenting a particular input image into several regions identified as articles, using this data it is then able to perform basic OCR. 

In the future, features such as title, author and content extraction would be a great addition to improving the utility of data generated. 

## Why is this tool needed

There has been a large effort put into digitizing old literary works, including historic newspapers. Although having digital copies of this data is step forward without a method for searching the data it significantly less accessible to the average person, academic researcher etc. 

Due to both the time required to manually label data and the sheer volume of data that exists an automatic method for segmenting and labeling data, extracting authors, titles, and content is required. This problem is not a new one either, there exists research to help solve this problem that attempts to solve this problem using a variety of approaches. 

**\<Add Sources>**

## Big Overview of Tool

The tool operates in three principal steps, all wrapped by a single command line entry point.

1. Input data is run through a DNN based on TensorFlow and images are labeled
2. Labeled images pass through a post-processing script that attempts to segment images using the labels from the previous step.
3. Using the segmentation data(could also be called bounding boxes) each identified article is run through OCR


## Getting Started

Here we provide a quick overview on how to get started using the project, processing data through etc. 

#### Cloning

This project contains a git submodule so you will need to initialize that in addition to cloning the project. See the [git book](https://git-scm.com/book/en/v2/Git-Tools-Submodules) for more info how submodules work.

* `git clone repo.url.git`
* In the cloned directory run:
    * `git submodule update --init --recursive`

#### Requirements
 - Python `3.7` (Python `>= 3.8` is not supported)
 - Tesseract `4.x`
 - Pipenv

#### Installing Python Requirements

Python dependencies are handled by Pipenv thus easy to install and keep updated. The Pipfile that is located in the root of this repo contains all required dependencies needed to run the project end-to-end -- including the ML model.

Simply run in the project directory:

* `pipenv install`
* `pipenv shell`
* `python main.py`

Congrats, you're not 100% setup to run the project!

#### External Dependencies

There are two principal external dependencies, the pre-trained model and `tesseract-ocr`. 

##### Pre-trained Model

To utilize the first part of the pipeline you will need the pre-trained model. As the ML model of this project is based on [bbz-segment](https://github.com/poke1024/bbz-segment) you can use their model which is provided [here on dropbox](https://www.dropbox.com/sh/7tph1tzscw3cb8r/AAA9WxhqoKJu9jLfVU5GqgkFa?dl=0). 

For convience, we also provide an archive of the complete model above [here](https://drive.google.com/file/d/1qNcZxpfqUGnsdy-V8vdo9G9wF1TBfXDK/view?usp=sharing).

In addition, [here is a link]() to just the separator model, the only one currently utilized in this version of the model. **This is the recommended model file to download**

Once downloaded, the model should be extracted to a convenient location. You will need to provide the path to the model to the CLI tool.


##### Tesseract

`tesseract` is the OCR engine used for the third step in the pipeline. It must be downloaded and or installed. Ultimately, you just need to know the path to the appropriate executable for your system. 

See the [tesseract-ocr documentation](https://tesseract-ocr.github.io/tessdoc/Installation.html) about installing the required files. Remember to keep the path of your tesseract install handy as you will need to provide it to `CLI` application.


#### Input File Requirements

Although this pipeline has been designed with next steps in mind it has been designed with the Liberator Dataset specifically, thus there are certain expectations for the input files, their naming format etc. 

##### Quick Overview

Input images must be JPGs of at least `2400,1200`(height, width). They must be named with the following convention: `issueid_imageid.jpg`

##### In depth Input Image Requirements

A given image in the Liberator dataset contains to key pieces of information: 
 * issue ID
 * image ID

 Both are globally unique and an image has both. That is to say, a given image has an ID and also belongs to a given issue(which typically contains some number of images comprising an entire image).

As such, both IDs are used extensively throughout the pipeline for identify images etc. Input files **must** has the following naming convention.

> issueid_imageid.jpg

Neither the issue ID or image ID can contain the special character that is an underscore because this is used to separate the two fields. 


In theory, the pipeline supports other file extensions, but for right now we're limiting the input dataset to jpeg format. Under the hood `Pillow` and `OpenCV` are used to read images and thus they should support other formats but they're currently untested.

Lastly, the input size must be at least `2400x1200` height, width. This is a soft lower limit that we have set for this version of the pipeline. It could possibly be changed in the future.


## Deep Dive

Here we hope to take a deep dive into each component of the pipeline, talk about the code a bit more specifically and provide insights into choices that where made and possible room for improvement.


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
- config.json file with ABBYY API ID, Password, and server url if you are using ABBYY OCR! [ABBYY OCR](https://www.ocrsdk.com/), also must have the code to run the ABBYY Online SDK downloaded and include the path to that and that config.json file when using it ***(ABBYY IS NOT SUPPORTED RIGHT NOW)***

Download and extract the model to `05_prediction/data/models`

Navigate to `bbz-segment` and run `pipenv install`

## Testing

Run inside a pipenv shell:

> `cd 05_prediction`

> `python src/main.py data/pages data/models`

Segmented images saved to `./data` root.

## Command Line Interface

As mentioned previously, the <strong>main.py</strong> is the overall wrapper for this program!
Ensure you have started the Pipenv environment with Python 3.7 and installed all dependencies PLUS installed external dependencies such as Tesseract, and the Model.
Once you have installed all dependencies you are ready to run the below commands:

<br/>

***Understanding all of the flags available:***
```bash
python main.py -h
```

<br/>

***Required Arguments:***
1. ***image_directory:*** the path to the directory containing all the images you would like to pass into this pipeline
2. ***image_extensions:*** the file format for the images that you are working in! **NOTE:** we only accept *.jpg* format as of right now
3. ***output_directory:*** the path to the directory that you will save the results to in each step of the pipeline, and also the input directory for the next stage in the pipeline! **NOTE:** this MUST be a full path! NOT a relative path
4. ***model_directory:*** the path to the directory containing the segmentation model! **NOTE:** inside this directory must be a folder named "*v3*" then inside that a directory named "*sep*" and finally inside that directories named "*1*", "*2*", "*3*", "*4*" and "*5*". This is how the model will be downloaded on your computer, so just don't mess with the directories it lives in

***Required Flags:***
1. ***-t:*** this is the Tesseract flag. You specify the path to the Tesseract executable on your computer here. ***NOTE:*** the directory where the tesseract executable file lives in should have all of it's dependencies as well, so DO NOT move the executable file away from it's dependencies.

***Example Usage:***
```bash
python main.py "./image_directory" "jpg" "<full_path>/output_directory" "./model" -t "D:/PyTesseract/tesseract.exe"
```

In the above command I had the directories "<strong>image_directory</strong>" and "<strong>model</strong>" in the same directory "<strong>main.py</strong>" was contained in. Regardless of where the "<strong>output_directory</strong>" directory is, you want to put in the full path to that directory. Finally, I wrote the required "<strong>-t</strong>" flag, and used the full path of where my tesseract executable is.

## Individual Pipeline

As of right now we have not included in the command line interface the ability to run an individual stage of the pipeline. A work around to this is to go into the "<strong>main.py</strong>" file and comment out pipelines you don't want to run in the "main()" method! Example Below:

<br/>

***Running All Pipelines:***
```python
### Step 1: Generate .npy file using bbz-segment and the model
bulk_generate_separators(args['image_directory'], args['image_extensions'], args['output_directory'], args['model_directory'], args['regenerate'], args['debug'], args['verbose'])

### Step 2: Get bounding boxes from .npy file   
segment_all_images(args['output_directory'], args['image_directory'], args['output_directory'], args['debug']) # TODO: When this directory and file exist uncomment this 

### Step 3: Run OCR on the generated bounding boxes
JSON_NAME = 'data.json' # NOTE: This is the name of the JSON file saved at Step 2 of the pipeline
json_path = os.path.join(args['output_directory'], JSON_NAME)
image_to_article_OCR(json_path, args['output_directory'], args['image_directory'], "tesseract")
```

<br/>

***Running Last 2 Pipelines:***
```python
### Step 1: Generate .npy file using bbz-segment and the model
# NOTE: I commented out Step 1 so the code will only run Steps 2 and 3 of the pipeline, but make sure the dependent files in Step 1 exist already
# bulk_generate_separators(args['image_directory'], args['image_extensions'], args['output_directory'], args['model_directory'], args['regenerate'], args['debug'], args['verbose'])

### Step 2: Get bounding boxes from .npy file   
segment_all_images(args['output_directory'], args['image_directory'], args['output_directory'], args['debug']) # TODO: When this directory and file exist uncomment this 

### Step 3: Run OCR on the generated bounding boxes
JSON_NAME = 'data.json' # NOTE: This is the name of the JSON file saved at Step 2 of the pipeline
json_path = os.path.join(args['output_directory'], JSON_NAME)
image_to_article_OCR(json_path, args['output_directory'], args['image_directory'], "tesseract")
```
 