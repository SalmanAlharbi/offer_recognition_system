# weekly-offers-detection-system
 ## Introduction
this project contains two models the way how they were trained the datasets they were trained which you can also find in this repo with the scripts used to create them and finally a streamlit website that deploys those two models into a pipeline of models that detect and recognize the weekly offers booklets of Saudi Major supermarkets specifically Panda and Altmimi booklets. Still, it can be used with other similar formatted booklets.


## main packages and libraries used
streamlit, cv2, tensorflow, pytesseract, spacy
<br />this repo contains the object_detection from tensorflow which can be found in <a href="https://github.com/tensorflow/models/tree/master/research/object_detection">this repo</a>

## Data
to see more details about the data and its structure go to <a href="https://github.com/Salma7577/offers_dataset_web_scrabing">this repo</a>

## Models 
### Offers detector model
this model is a traiend using SSD MobileNet V2 FPNLite 320x320 pretreained model it takes as an input a booklet page like the one bellow.

<img src="https://user-images.githubusercontent.com/54520739/186296014-976bef57-21fe-4cdc-9110-c0040f76043b.jpg" width="600" height="800">


and outputs a bounding box for each offers on the image.

<img src="https://user-images.githubusercontent.com/54520739/186297684-77e80c6d-ea93-4664-b892-a1d1a21fb977.png" width="600" height="800">

### offers' data detector model
this model is also traiend using SSD MobileNet V2 FPNLite 320x320 pretreained model but it takes the bonding boxes from the erlier model "one a time".


<img src="https://user-images.githubusercontent.com/54520739/186298696-73ab7204-c45b-4db6-9bb3-d2d28a6b10ac.jpg" width="250" height="250">
and outputs a bounding box for each information of the offer bounding box.


<img src="https://user-images.githubusercontent.com/54520739/186298937-09ee2266-fcfd-4985-b22b-7fdfdde50936.png" width="250" height="250">



## repo structure
```bash
├───assets
│   └───image_examples
├───bin #reqiered to run the model
├───my_models
│   └───models
│       ├───offer_box_detection_model
│       └───offer_data_recognition
├───protoc #reqiered to run the model
├───src  #reqiered to run the model
└───workspace
    ├───annotations
    ├───images
    │   ├───collectedimages
    │   │   ├───offers_only_dataset
    │   │   └───offer_box_dataset
    ├───models
    │   ├───offer_box_detection_model
    │   └───offer_data_recognition
    └───pretrained_models
        ├───ssd_mobilenet_v2_fpnlite_320x320


```


## to improve in the future


