# 1. Table Of Contents
- [1. Table Of Contents](#1-table-of-contents)
- [2. Newspaper Article Segmentation](#2-newspaper-article-segmentation)
- [3. Why is this tool needed](#3-why-is-this-tool-needed)
- [4. End goal of this tool](#4-end-goal-of-this-tool)
  - [4.1. Current State](#41-current-state)
  - [4.2. Next Steps](#42-next-steps)
- [5. Big Overview of Tool](#5-big-overview-of-tool)
- [6. Getting Started](#6-getting-started)
  - [6.1. Cloning](#61-cloning)
  - [6.2. Requirements](#62-requirements)
  - [6.3. Installing Python Requirements](#63-installing-python-requirements)
  - [6.4. External Dependencies](#64-external-dependencies)
  - [6.5. Pre-trained Model](#65-pre-trained-model)
  - [6.6. Tesseract](#66-tesseract)
  - [6.7. libGL](#67-libgl)
- [7. Input File Requirements](#7-input-file-requirements)
  - [7.1. Quick Overview](#71-quick-overview)
  - [7.2. In depth Input Image Requirements](#72-in-depth-input-image-requirements)
- [8. Command Line Interface](#8-command-line-interface)
  - [8.1. Individual Pipeline](#81-individual-pipeline)
- [9. Pipeline Architecture](#9-pipeline-architecture)
- [10. Deep Dive Into The Pipeline](#10-deep-dive-into-the-pipeline)
  - [10.1. Deep Dive: ML Model, Image Labeling](#101-deep-dive-ml-model-image-labeling)
    - [10.1.1. Model Overview](#1011-model-overview)
    - [10.1.2. End-To-End Model Function](#1012-end-to-end-model-function)
  - [10.2. Deep Dive: Article Segmentation](#102-deep-dive-article-segmentation)
    - [10.2.1. Speed Considerations](#1021-speed-considerations)
  - [10.3. Deep Dive: Content Extraction(OCR)](#103-deep-dive-content-extractionocr)
- [11. Dataset and Sample Results](#11-dataset-and-sample-results)
  - [11.1. Complete Dataset](#111-complete-dataset)
  - [11.2. Sample Dataset & Results](#112-sample-dataset--results)
- [12. Alternative Methods Tried](#12-alternative-methods-tried)
- [13. Extra Resources](#13-extra-resources)

# 2. Newspaper Article Segmentation

This repository contains tools for segmenting archives of newspapers, in the form of digital images into individual articles containing various article level information, such as content, author, title etc. It utilizes a DNN and a set of post-processing scripts to segment images into articles, producing a JSON output that contains the detected information. 

Currently, it supports segmenting a particular input image into several regions identified as articles, using this data it is then able to perform basic OCR. 

In the future, features such as title, author and content extraction would be a great addition to improving the utility of data generated. 

# 3. Why is this tool needed

There has been a large effort put into digitizing old literary works, including historic newspapers. Although having digital copies of this data is step forward without a method for searching the data it significantly less accessible to the average person, academic researcher etc. 

Due to both the time required to manually label data and the sheer volume of data that exists an automatic method for segmenting and labeling data, extracting authors, titles, and content is required. This problem is not a new one either, there exists research to help solve this problem that attempts to solve this problem using a variety of approaches. 

**\<Add Sources>**

# 4. End goal of this tool

The "ideal" version of this tool would be able to take as input a set of images that represent a number of historical newspaper issues. Segment those images into individual issues, extracting the location of each article on the image. Then, among each article extracting:

* Article Content
* Article Title
* Article Author
* Published Date(If applicable)

Further, providing a set of topics/keywords about each article to aid in search. As in, performing some sort of topic analysis on the extracted text to aid in classification of individual articles and issues.

## 4.1. Current State

With the end goal in mind, the tool as of right now is able to:

 
* Extract article location (with about 70% accuracy)
* Perform basic OCR on detected regions.

Clearly, there are several points left to complete and the article location extraction needs some polishing to improve the accuracy, especially on more "difficult" to read archives(partially damaged image, more worn page etc).


## 4.2. Next Steps

It cannot be stressed enough that this is an alpha version of a tool. It needs much refining before it would be able to produce data that would be useful to the archive systems. We have identfied the following as possible next steps for the next person(s) who want to work on this project. 

* Train model on data in the dataset, ie, manually label some set of images(20-50) and perform some transfer learning. 

* Train a model(this or another) to detect the titles of articles, in attempt to help extract this data.

* Optimize the method that is used to take the labeled image and generated a list of polygons that represent article regions. It is currently quite slow and just generally under performs.

* Perform content/topic analysis on extracted text 


# 5. Big Overview of Tool

The tool operates in three principal steps, all wrapped by a single command line entry point.

1. Input data is run through a DNN based on TensorFlow and images are labeled
2. Labeled images pass through a post-processing script that attempts to segment images using the labels from the previous step.
3. Using the segmentation data(could also be called bounding boxes) each identified article is run through OCR


# 6. Getting Started

Here we provide a quick overview on how to get started using the project, processing data through etc. 

## 6.1. Cloning

This project contains a git submodule so you will need to initialize that in addition to cloning the project. See the [git book](https://git-scm.com/book/en/v2/Git-Tools-Submodules) for more info how submodules work.

* `git clone repo.url.git`
* In the cloned directory run:
    * `git submodule update --init --recursive`

## 6.2. Requirements
 - Python `3.7` (Python `>= 3.8` is not supported)
 - Tesseract `4.x`
 - Pipenv

## 6.3. Installing Python Requirements

Python dependencies are handled by Pipenv thus easy to install and keep updated. The Pipfile that is located in the root of this repo contains all required dependencies needed to run the project end-to-end -- including the ML model.

Simply run in the project directory:

* `pipenv install`
* `pipenv shell`
* `python main.py`

Congrats, you're not 100% setup to run the project!

## 6.4. External Dependencies

There are three principal external dependencies, the pre-trained model, `tesseract-ocr` and libgl(Linux specific). 

## 6.5. Pre-trained Model

To utilize the first part of the pipeline you will need the pre-trained model. As the ML model of this project is based on [bbz-segment](https://github.com/poke1024/bbz-segment) you can use their model which is provided [here on dropbox](https://www.dropbox.com/sh/7tph1tzscw3cb8r/AAA9WxhqoKJu9jLfVU5GqgkFa?dl=0). 

For convience, we also provide an archive of the complete model above [here](https://drive.google.com/file/d/1qNcZxpfqUGnsdy-V8vdo9G9wF1TBfXDK/view?usp=sharing).

In addition, [here is a link]() to just the separator model, the only one currently utilized in this version of the model. **This is the recommended model file to download**

Once downloaded, the model should be extracted to a convenient location. You will need to provide the path to the model to the CLI tool.


## 6.6. Tesseract

`tesseract` is the OCR engine used for the third step in the pipeline. It must be downloaded and or installed. Ultimately, you just need to know the path to the appropriate executable for your system. 

See the [tesseract-ocr documentation](https://tesseract-ocr.github.io/tessdoc/Installation.html) about installing the required files. Remember to keep the path of your tesseract install handy as you will need to provide it to `CLI` application.

## 6.7. libGL

Most unix systems require libGL for opencv, for Ubuntu:

> `sudo apt-get install libgl1-mesa-glx`




# 7. Input File Requirements

Although this pipeline has been designed with next steps in mind it has been designed with the Liberator Dataset specifically, thus there are certain expectations for the input files, their naming format etc. 

## 7.1. Quick Overview

Input images must be JPGs of at least `2400,1200`(height, width). They must be named with the following convention: `issueid_imageid.jpg`

## 7.2. In depth Input Image Requirements

A given image in the Liberator dataset contains to key pieces of information: 
 * issue ID
 * image ID

 Both are globally unique and an image has both. That is to say, a given image has an ID and also belongs to a given issue(which typically contains some number of images comprising an entire image).

As such, both IDs are used extensively throughout the pipeline for identify images etc. Input files **must** has the following naming convention.

> issueid_imageid.jpg

Neither the issue ID or image ID can contain the special character that is an underscore because this is used to separate the two fields. 


In theory, the pipeline supports other file extensions, but for right now we're limiting the input dataset to jpeg format. Under the hood `Pillow` and `OpenCV` are used to read images and thus they should support other formats but they're currently untested.

Lastly, the input size must be at least `2400x1200` height, width. This is a soft lower limit that we have set for this version of the pipeline. It could possibly be changed in the future.


# 8. Command Line Interface

As mentioned previously, the `main.py` is the overall wrapper for this program!
Ensure you have started the Pipenv environment with Python 3.7 and installed all dependencies PLUS installed external dependencies such as Tesseract, and the Model.
Once you have installed all dependencies you are ready to run the below commands:

<br/>

***Understanding all of the flags available:***
```bash
python main.py -h
```

<br/>

***Required Arguments:***
1. ***`image_directory`:*** the path to the directory containing all the images you would like to pass into this pipeline
2. ***`image_extensions`:*** the file format for the images that you are working in! **NOTE:** we only accept *.jpg* format as of right now
3. ***`output_directory`:*** the path to the directory that you will save the results to in each step of the pipeline, and also the input directory for the next stage in the pipeline! **NOTE:** this MUST be a full path! NOT a relative path
4. ***`model_directory`:*** the path to the directory containing the segmentation model! **NOTE:** inside this directory must be a folder named "*v3*" then inside that a directory named "*sep*" and finally inside that directories named "*1*", "*2*", "*3*", "*4*" and "*5*". This is how the model will be downloaded on your computer, so just don't mess with the directories it lives in

***Required Flags:***
1. ***`-t`:*** this is the Tesseract flag. You specify the path to the Tesseract executable on your computer here. ***NOTE:*** the directory where the tesseract executable file lives in should have all of it's dependencies as well, so DO NOT move the executable file away from it's dependencies.

***Example Usage:***
```bash
python main.py "./image_directory" "jpg" "<full_path>/output_directory" "./model" -t "D:/PyTesseract/tesseract.exe"
```

In the above command I had the directories `image_directory` and `model` in the same directory `main.py` was contained in. Regardless of where the `output_directory` directory is, you want to put in the full path to that directory. Finally, I wrote the required `-t` flag, and used the full path of where my tesseract executable is.

## 8.1. Individual Pipeline

As of right now we have not included in the command line interface the ability to run an individual stage of the pipeline. A work around to this is to go into `main.py` file and comment out pipelines you don't want to run in the `main()` method! Example Below:

<br/>

The following are excerpts from `main.py`

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

# 9. Pipeline Architecture

The pipeline was designed in such a manner that enables independent development of each part and or addition of new steps. Given that the output data from each step is saved sub-sections of the pipeline can be re-run without necessitating the re-run of the entire pipeline. 

For example, if one wanted to improve the article region detection algorithm(that leverages the output from the ML model) it would not be needed to re-run the labelling model since those files have already been generated. 

Outside of the first step, the ML model that labels images, the common exchange format between each step is the JSON file containing the detected information about the articles/issues. If a need feature is desired to be added the author should try as much as possible to continue this design choice. A new "step" in the pipeline should read the input from the previous steps and then perform the new operations, outputting a **new** `JSON` file for use by the next step of the pipeline. Always preserving both it's input and output.


# 10. Deep Dive Into The Pipeline

Here we hope to take a deep dive into each component of the pipeline, talk about the code a bit more specifically and provide insights into choices that where made and possible room for improvement.

## 10.1. Deep Dive: ML Model, Image Labeling

// TODO: Update the name

This part of the code handles the labeling of images using the DNN model. This model is based on other research work, thus this part of the code has been kept in an open-source location to comply with the license. 

The original work can be found at [this github repository](https://github.com/poke1024/bbz-segment) and [published in this paper](https://arxiv.org/abs/2004.07317).

### 10.1.1. Model Overview

The model is actually comprised of two distinct parts, which are called `sep` and `blkx` in the original repository and academic paper. The first one `sep` attempts to segment a given image into the following components: background, horizontal separator, vertical separator, and table separators. The 2nd, `blkx` attempts to classify regions of the image into one of the following categories: background, text, tabular data, illustrations. 

In this pipeline we only use the first one, `sep` of which we only utilize the identified vertical and horizontal separators to help segment the images. 

In both types of models, the model performs a pixel level labeling of the image. That is, each pixel in the image is assigned a particular label, an integer value. In addition, images are resized to `2400x1200` before processing.  

For more in depth information it would be useful to read the above linked paper.


### 10.1.2. End-To-End Model Function

This section describes the general overview of how an image is taken from it's original input and passed through the ML model, finally outputting the pixel level labels.

First, input images are resized to `2400x1200`, this sized is a property that has been set in the pre-trained model and is loaded from that model file at runtime. Images are then broken into smaller tiles, this allows the processing of images without as high of video memory requirements. 

After passing through the model the output is a numpy array, where each `(row, col)` pair represents a pixel in the given resized image and the label assigned by the model. Depending on the command line options used this data is saved into the following format:

Always saved(As a Pickle via numpy):
* Labeled numpy array
* Image original size
* Input filename

If a debug parameter was passed the following is also saved:
* A reproduction of the labeled image, where each label gets it's own color. This allows the inspection of how the model labeled the input image. 


This information is then saved for the next part of the pipeline. 

## 10.2. Deep Dive: Article Segmentation

This step of the pipeline attempts to utilize the data from the previous step to create a set of polygons that represent the detected article regions of the input image. There is done via a simple rule based algorithm. It is exclusively utilizes the detected horizontal and vertical separators to make these detections.

The output from this step is a partially filed `JSON` file in the form as defined in the output section of this document. It contains a list of `articles`, each with a set of images and polygons that makeup that article. At this step, no attempts to join articles across multiple pages has been made. Thus, each article object only contains a single image, although it may contain multiple polygons within a single image. 

This step also outputs a debug image, if specified at invocation. This image contains the original image annotated with colored boxes drawn around each identified article region. This allows inspection of how the algorithm chose individual articles. 

**Check the to-do list for current issues with this step**

### 10.2.1. Speed Considerations

Despite the reasonably simplistic nature of this step it takes a considerable amount of time to run. Typically about 30-45s per image. No attempt to improve this speed has been attempted, the easiest would be to process multiple images in parallel, but there does exist opportunities for improving the speed of the pipeline.

## 10.3. Deep Dive: Content Extraction(OCR)

This step of the pipeline takes as input the partially filed JSON file from the previous step and uses the detected article regions and performs OCR, filling the JSON file with the detected text. 

Currently this module uses `tesseract-ocr` with default settings to perform this detection of text. In addition, no attempt at detecting keywords, authors, article titles etc. is made. 

# 11. Dataset and Sample Results

We provide links to all data utilizes for this project. Specific resources are broken out below but here is the link to the entire Google Drive folder:

[Folder with Data](https://drive.google.com/drive/folders/1Kc9K_lCCHqfHi-FopoN5oqzfAp515Ghz?usp=sharing)

## 11.1. Complete Dataset
The complete dataset can be downloaded from Google Drive here, or re-scraped using the `data_downloader` tool if the latest data is required. One could also jump-start the scraping by downloading our archived dataset and re-running the data-downloader which will only download images not in the dataset. 

[Complete Dataset](https://drive.google.com/file/d/1xbcfguni-1j0uRPfVBPlSILDU8dmXATa/view?usp=sharing)

[CSV with image links and metadata](https://drive.google.com/file/d/1L5IBr9tGysHQLx0vEGDimnS3MJ8ghISz/view?usp=sharing)


## 11.2. Sample Dataset & Results

We also provide the sample dataset which contains 25 issues comprised of 100 images. This is our test data and contains annotations for the number of ground-truth articles per issue. It also contains the output from the current version of the model, including debug data(annotated images etc), article segmentation and OCR results. 

[Sample Images Only](https://drive.google.com/file/d/1b4zuJ2yr3--shQJOBFIXbufSlJL4pelN/view?usp=sharing)

[Sample Images w/Output at each step]()




# 12. Alternative Methods Tried

 We briefly explored other methods for accomplishing this task, mainly AWS Textract. Although textract tends to do very well with structured data, we did not see a good performance on our dataset. About the only thing it could infer was *rough* column/row data. These divisions often did not indicate the actual location of columns/rows and did not do as good of a job as the model we utilized. 

 # 13. Extra Resources

 Here we provide a list of extra resources that anyone else working on this project might find helpful. 

  <br/>

**[An Auto-Encoder Strategy for Adaptive Image Segmentation]( <br/>
)**:

 This paper presents a method that utilizes an auto-encoder to segment MRI images. It's two main features are it's ability to learn with hardly any reference data(1 labeled image). In addition, it is much more computationally efficient than other approaches. Code is available on github. 

  <br/>

 **[Combining Visual and Textual Features for Semantic Segmentation of Historical Newspapers](https://arxiv.org/abs/2002.06144)**

 Similar approach and model to the paper we based our work on. Code also available on github. 
   <br/>


 **[Clustering-Based Article Identification in Historical Newspapers](https://www.aclweb.org/anthology/W19-2502/)**
 
 This paper explores the method of utilizing NLP approaches to segmenting articles, finding reading order etc. Their method takes the OCR data and utilizes various NLP approaches in attempt to identify where one article starts or end. 
   <br/>

  <br/>

 **[Logical segmentation for article extraction in digitized old newspapers](https://dl.acm.org/doi/10.1145/2361354.2361383)**

  <br/>

 **[Fully Convolutional Neural Networks for Newspaper Article Segmentation](https://ieeexplore.ieee.org/document/8270006)**

