# Fake-news-catcher
Fake News Detection using Logistic Regression and TF-IDF on real-world datasets.
# Fake News Detection using Machine Learning

This project is a machine learning-based system designed to detect whether a news article is real or fake. It uses Logistic Regression along with TF-IDF vectorization to classify news content based on patterns learned from real-world datasets.

The model is trained on a combination of fake and genuine news articles, allowing it to distinguish between reliable and misleading information with high accuracy.

## Features

* Uses real-world datasets (Fake.csv and True.csv)
* Text preprocessing using regular expressions
* TF-IDF feature extraction for numerical representation of text
* Logistic Regression model for classification
* Accuracy evaluation of the trained model
* Confusion matrix visualization for performance analysis
* Interactive user input for real-time predictions

## Dataset

The dataset used in this project consists of two parts:

* **Fake.csv**: Contains fake news articles
* **True.csv**: Contains real news articles

Both datasets are combined and labeled appropriately before training.

## How It Works

The workflow of the project is as follows:

1. The datasets are loaded and combined into a single dataset
2. Labels are assigned (0 for fake news and 1 for real news)
3. The text data is cleaned and preprocessed
4. TF-IDF vectorization converts text into numerical features
5. The Logistic Regression model is trained on the processed data
6. The model is evaluated using accuracy and a confusion matrix
7. The user can input custom news text for prediction

## Model Performance

The model achieves an accuracy of approximately **98–99%** on the test dataset. This indicates strong performance on the given data.

However, accuracy alone does not fully determine real-world effectiveness, and further validation is necessary for broader applications.

## Limitations

This model performs best on longer, well-structured news articles similar to those in the training dataset. It may not perform accurately on very short inputs such as headlines, informal statements, or incomplete sentences.

Since the model relies on TF-IDF features, it captures word frequency patterns rather than deep contextual meaning. This limits its ability to fully understand nuanced or ambiguous content. As a result, predictions on short or unconventional text may be less reliable.

## How to Run

1. Install the required libraries:

pip install -r requirements.txt

2. Run the program:

python main.py

3. Enter news text when prompted to get predictions.

## Future Improvements

* Use deep learning models such as BERT for better contextual understanding
* Improve performance on short and informal text inputs
* Build a web-based interface for easier interaction
* Expand dataset to include more diverse sources

## Conclusion

This project demonstrates how machine learning can be applied to detect fake news using text classification techniques. While the model performs well on structured data, further improvements can enhance its real-world reliability and robustness.
