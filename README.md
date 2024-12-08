# Emotion-Detection

This project focuses on detecting human emotions from facial expressions using deep learning. The system is trained on a labeled dataset of facial images and predicts emotions such as happy, sad, angry, surprised, and more.

The application can be used in various fields, including:

* Customer sentiment analysis
* Mental health monitoring
* Human-computer interaction
  ![emo](https://github.com/user-attachments/assets/776c9f1e-ba17-4e48-bb73-6f97a6b939c0)<br />
  *Image Source: [Aratek - How Does Facial Emotion Recognition Express Your Feelings?](https://www.aratek.co/news/how-does-facial-emotion-recognition-express-your-feelings)*

## Dataset
The project uses a publicly available emotion dataset, such as FER2013. You can download it from https://www.kaggle.com/datasets/msambare/fer2013.

## Features
* Real-time Emotion Detection: Detect emotions using live video feeds 
* Multi-emotion Classification: Supports detection of multiple emotions (e.g., angry, happy, neutral, etc.).
* User-Friendly Interface: Easy-to-use interface for  enabling camera feeds.
## Installation
Follow these steps to set up the repository: <br />
### Clone the repository:
``` bash
git clone git@github.com:Ravinderram/Emotion-Detection-.git
```
### Install dependencies:
``` bash
pip install -r requirements.txt
```
## Usage 
1. First run the below ```.py``` file and save the model. Don't forgot to give the dataset path.
``` bash
emotion_detection traning file.ipynb
```
2. Next run the given ```.py``` file and give saved model and xml file  path into the main  file.
``` bash
emotion_detection traning file.ipynb
```
``` bash
haarcascade_frontalface_default.xml
```
## Example 
![dfg](https://github.com/user-attachments/assets/630e593a-1367-462b-ab4c-ffb279cbd993)
