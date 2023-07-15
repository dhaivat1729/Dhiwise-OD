# Dhiwise object detection 

## Table of Contents
- [Dhiwise object detection](#dhiwise-object-detection)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
  - [Directory Structure](#directory-structure)
  - [Annotation file structure](#annotation-file-structure)
  - [Training](#training)
  - [Evaluation](#evaluation)
  - [Inference](#inference)
   

## Installation
Installation script is provided to install all the dependencies. It will create a virtual environment and install all the dependencies in it. 

```bash
chmod +x setup.sh
./setup.sh
```

## Directory Structure
Dataset directoru should be in the following format:

```
/path/to/dataset
├── images
│   ├── <image0>.jpg
│   ├── <image1>.jpg
│   ├── ...
│   └── <imageN>.jpg
├── annotations.json
```

## Annotation file structure
annotations.json should be in the following json format.
```json
[
    {
        "screenName": "<image0>.jpg",
        "class" : "class0",
        "children": [
            {
                "width" : 100,
                "height" : 100,
                "x" : 100,
                "y" : 100
                "type": "classLabel"
            }
            ...
        ]
        "screenName": "<image1>.jpg",
        "class" : "ImageClass",
        "children": [
            {
                "width" : 100,
                "height" : 100,
                "x" : 100,
                "y" : 100
                "type": "classLabel"
            }
            .
            .
            .
        ]
    }
]

```

## Training
Training the model is a single line command. This codebase is built on top of detectron2. So, all the options available in detectron2 are available here too. check `/DWODLib/config/defaults.py` to understand configuration options. 


If you want to train an object detection model with resnet 50, you can run the following command. 

```python
python3 tools/train.py --options --outputDir "path/to/experiment/directory" --dataDir "/path/to/dataset" --batchSize 4 --numSteps 45000 --save_after_steps 1500 --bbox_loss "giou" --annotationFile "annotations.json" --detectorName "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
```

In the above command, outputDir is created during training. So it need not exist before training. If that directory already exists, you will be prompted to delete it before prroceeding with training.
Other options can be configured through command line arguments. For example, if you want to set the learning rate to 1e-3, you can pass `--learningRate "1e-3"` to the train command. 

## Evaluation
Evaluation script is provided to evaluate the model on the validation set. 

```python
python3 tools/evaluation.py --options --outputDir '/path/to/experiment/directory' --dataSplit "val" --testScoreThresh 0.03
```
Through this script, you can run evaluation on train/validation scripts. Evaluation on arbitrary testset is not supported yet. 

## Inference
Requirement: Images you want to run inference on should be located in a particular directory. Path to where images should be stored should be provided too, the script will automatically create that new directory if it doesn't exist. 

```python
python3 tools/inference.py --options --inputImagesDir '/path/to/input/images' --modelPathDirectory '/path/to/experiment/directory' --outputImageDir '/path/to/output/images'
```

