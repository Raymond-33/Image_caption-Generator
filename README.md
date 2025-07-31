
Image Caption Generator using Deep Learning

Overview:

This project builds an end-to-end deep learning model that automatically generates descriptive captions for input images. It combines the power of Convolutional Neural Networks (CNN) for image feature extraction and Long Short-Term Memory (LSTM) networks for sequence generation. The model is trained on the Flickr8k dataset and generates captions that resemble human-written descriptions.

Key Features:

* Accepts image as input and generates relevant captions
* Utilizes CNN (InceptionV3 or VGG16) for feature extraction
* Uses LSTM layers for natural language generation
* Achieves BLEU score of 0.58 on test dataset
* Trained on over 8,000 images with five captions each
* Optimized for T4 GPU and batch training

Technologies Used:

* Python
* TensorFlow / Keras
* NumPy, Pandas
* NLTK (for BLEU score evaluation)
* Matplotlib (for plotting training loss)
* Google Colab (GPU-accelerated training)

Project Workflow:

1. Extract features from images using a pretrained CNN
2. Preprocess and tokenize the captions
3. Map image features with tokenized text
4. Train a combined CNN + LSTM model on paired data
5. Generate captions for new, unseen images
6. Display image and generated caption together

How to Use:

1. Clone the repository or download project files
2. Download the required dataset and model files from the link below:
   **Google Drive Link:** \[Insert your link here]
3. Upload an image into the interface or script environment
4. Run `generate_caption.py` or use the terminal interface to get the output
5. The system will display the image along with the predicted caption

Project Output:

* BLEU Score: 0.58
* Sample Output:
  Input Image → "A man riding a bike on a dirt road."
  Input Image → "Two children are playing with a soccer ball."

Applications:

* Assistive tools for visually impaired individuals
* Automated image indexing and tagging
* AI-based content creation and storytelling
* Smart surveillance and social media platforms

Setup Instructions:

* Install required libraries:
  pip install tensorflow numpy pandas matplotlib nltk
* Download the pretrained model file (`caption_model.h5`) and tokenizer files from Google Drive
* Run the model on Google Colab or locally with GPU support for better performance
