## image-captioning

# Image Captioning with CNN-RNN Architecture

This repository presents a powerful image captioning system that automatically generates natural language descriptions for images. The system combines the strengths of computer vision and natural language processing using a classic encoder-decoder architecture.

-----

## **Project Overview**

The core of this system is an **encoder-decoder** framework. The **encoder**, a Convolutional Neural Network (CNN), processes an input image and extracts its key visual features. These features are then passed to the **decoder**, a Recurrent Neural Network (RNN), which acts as a language model, generating a caption word-by-word based on the visual context provided by the encoder.

This implementation leverages pre-trained CNN models like **ResNet** and **MobileNetV2** for robust feature extraction and uses **LSTM/GRU** networks as the decoder to handle sequential text generation. The project pipeline includes data preprocessing, model training, and evaluation using standard metrics.

-----

## **Key Features**

  * **Encoder-Decoder Architecture**: A seamless integration of a CNN encoder and an RNN decoder for end-to-end image captioning.
  * **Diverse CNN Backbones**: Supports multiple pre-trained CNN models (ResNet-18, ResNet-50, MobileNetV2) to allow for a trade-off between performance and computational efficiency.
  * **Efficient Data Handling**: The system includes utilities to download, preprocess, and load the Flickr8k dataset, handling tasks like tokenization and vocabulary creation.
  * **Flexible Training Pipeline**: The training loop is implemented with features like teacher forcing to stabilize the training process.
  * **Greedy Decoding**: Implements a simple yet effective greedy decoding strategy for generating captions during inference.
  * **Performance Evaluation**: Includes scripts for calculating standard metrics like **BLEU scores** to quantitatively evaluate the model's performance.

-----



## **Getting Started**

### **Installation**

1.  Clone the repository:

2.  Create a virtual environment and install the dependencies:

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    pip install -r requirements.txt
    ```

### **Dataset**

The project uses the **Flickr8k dataset**. Run the following command to automatically download and prepare the dataset:

```bash
python data/download_flickr.py
```

### **Usage**

The project is structured around a series of Jupyter notebooks that guide you through the entire process, from data exploration to model evaluation.

1.  **Data Exploration**: Begin with `1_Data_Exploration.ipynb` to understand the dataset's characteristics, vocabulary size, and caption length distribution.
2.  **Feature Extraction**: Use `2_Feature_Extraction.ipynb` to extract image features from a pre-trained CNN and save them for efficient training.
3.  **Model Training**: Train the encoder-decoder model by running `3_Model_Training.ipynb`. This notebook handles the training loop, loss calculation, and model checkpointing.
4.  **Evaluation**: Finally, `4_Evaluation_Visualization.ipynb` allows you to generate captions for test images, calculate performance metrics, and visualize the results.

-----

## **Example Captions**

After training, the model can generate descriptive captions for various images. Here are a few examples of what the system can produce:

  * "A brown dog is running through the grass."
  * "A man in a red shirt is climbing a rock wall."
  * "Children are playing soccer on a green field."
