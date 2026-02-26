# Image & Text consistency - Project
<img src="https://github.com/Se-Boruk/Image_Text_Consistency_Project/blob/master/Assets/Dog_img.png?raw=true" width="520">

This project was part of the Artificial Intelligence & Machine Learning course.

It focuses on creating method for processing image and text inputs and telling if the text description is consistent with the image content. 

In other words: Is the text describing the image accurately?

## Project contains the files:
- **Config.py**:
  > Most important training parameters
- **Main_train.py**:
  > Training loop. It contains all logic related to training the model, with minor hyperparameters related to saving frequency or patience
- **Architectures.py**:
  > Contains the architectures of used Deep Learning models (CNN + LSTM)
- **Tokenizer_lib.py**:
  > Library covering tokenizer class, as well as the creation of the vocabulary used later in the training and inference of the model
- **Evaluate_test.py**:
  > Script which evaluates the model performance on the test set as well as it adjust the optimal threshold value for decision. - It's little cheaty, however the "real" test set was hidden during the project duration by the project supervisor. The end test set has never seen the model before.
- **Database_create.py**:
  > Creation of the database from raw format to the processed .arrow database.
- **DataBase_functions.py**:
  > Script containing dataloaders and all necessary logic for handling the data loading, augmenting etc. during training
- **Negative_map.py**:
  > Map necessary for the logic of negative example creation. The database contained only positive examples.
- **Functions.py**:
  > Other functions used across the whole project
- **Analyze_logs.py**:
  > Script to visualize the logs gathered during the training. Plotting script

#### There are other files in the "Miscellaneous_scripts" folder. They are less important and are described in dedicated Miscellaneous_info.txt file


### Due to the file size, project does not contains:
- Training dataset
- Models: They are however avaliable to download inside the "Models" folder. Downloaded zip file has all necessary files to use the model standalone

## Used data & Processing

To train the model the coco, flickr30k and ade20k datasets has been used (train parts). 

The captions used were the standard ones (coco & flickr30k) but additional ones has been added such as "localized narratives" and "ShareGPT4V / Share-Captioner Dataset"

### **To create negative text examples 2 approach method were used:**<br>

First and basic one is just using other examples in the batch as negative samples. It is technique known as contrastive learning.

#### Contrastive learning example. Loss function makes dot products of diagonal (positive samples) bigger, while making the rest of the matrix dot products smaller
<img src="https://github.com/Se-Boruk/Image_Text_Consistency_Project/blob/master/Assets/Contrastive_learnin_example.png?raw=true" width="520">

Second one involved algorithmical creation of the negative samples by swapping words in the sentence.<br>
It has been done this way to ensure that the model is also sensitive to small mismatches in the descriptions, e.g.<br>
"There is **red** apple laying on the table" != "There is **green** apple laying on the table"<br>

To ensure correct and diverse word swaps, the "spacy" module was used. It is NLP tool which allows for basic identification and manipulation on text data. It's excellent for lightweight analysis and word swaps while ensuring we are swapping only matching context words.<br>

So it prevents f.e. situation where we would swap the word "watch" for the "bracelet" in the sentence: "There is a family who watch TV, sitting on the couch".

In general predefined categories of words have been swapped (colors, objects, genders...) in the sentence creating these hard negative examples. Full list of words and the logic is avalialbe in Negative_map.py

### **Image augmentation**<br>

Also the images has been augmented to generalize model training.<br>
Used techniques involved color jittering, cropping, grayscaling or blurring.

### It is important to note that both text negative examples and image augmentation were done dynamically during the training

#### Examples of produced training samples
<img src="https://github.com/Se-Boruk/Image_Text_Consistency_Project/blob/master/Assets/Data_visualisation_1.png?raw=true" width="640">


## Architectures & Training

### Architecture
Final Model is variation of Siameese Network.<br>
In one branch it has CNN (with ResNet50 pretrained backbone) and on the other one there is modified LSTM.

Both of the network are producing vectors containing informations about the text and image contents. Then the cosine similarity is calculated to determine if the text describes the image well. If it is above given threshold, then 2 samples are classified as match. If not then as mismatch.

#### Architecture scheme
<img src="https://github.com/Se-Boruk/Image_Text_Consistency_Project/blob/master/Assets/Contrastive_learnin_example.png?raw=true" width="520">

### Training course
Training used already combined triplet loss function (for hard negatives) and contrastive loss (for general knowledge).

The training techniques used include half-precision to reduce computational overhead, gradient accumulation for artificially increasing batch size (important for contrastive loss)

#### Training over the epochs. Metrics included
<img src="https://github.com/Se-Boruk/Image_Text_Consistency_Project/blob/master/Plots/metrics_over_training.png?raw=true" width="640">

At the end of the training the analysis of the best threshold has been performed on isolated local test set, to prepare model for the real test set.

#### Best threshold analysis
<img src="https://github.com/Se-Boruk/Image_Text_Consistency_Project/blob/master/Plots/Threshold_Analysis_best_model.png?raw=true" width="640">

## Results

To describe final results - what scores the model achieved and overal conclusion :)









