# image_classification
Classification of binary images corresponding to cutting edges of insert tools used in milling processes using 10 extracted features and several ML models. 

### Requirements
```
pip install -r requirements.txt
```
### Usage
The first file _extract_features.py_ should be launched first to extract the features for the dataset. This step is already done and the results are in the features are in the _Output_ folder.<br>
The seconde file _main.py_ is the classifier based on the extracted features, for further details check the code.<br>
### Dataset
The full dataset used for extracting the features can be found here on kaggle: [Dataset](https://www.kaggle.com/datasets/ayoubjhabli/image-classification).
