# weekly-offers-detection-system
 ## Introduction
this project contains two models the way how they were trained the datasets they were trained which you can also find in this repo with the scripts used to create them and finally a streamlit website that deploys those two models into a pipeline of models that detect and recognize the weekly offers booklets of Saudi Major supermarkets specifically Panda and Altmimi booklets. Still, it can be used with other similar formatted booklets.


## Models 
### Offers detector model
this models takes as an input a booklet page like the one bellow.
![img94](https://user-images.githubusercontent.com/54520739/186296014-976bef57-21fe-4cdc-9110-c0040f76043b.jpg)
<img src="https://user-images.githubusercontent.com/54520739/186296014-976bef57-21fe-4cdc-9110-c0040f76043b.jpg" width="100" height="160">
and outputs a bounding box for each offers on the image.
