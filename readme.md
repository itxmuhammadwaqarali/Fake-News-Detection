# Fake News Detection: A Text-Based Approach

### Official Repository for "Text-Based Fake News Detection Project"

This repository contains the code and resources for the **Text-Based Fake News Detection** project, focusing on identifying misinformation in news articles using NLP and machine learning techniques.

## Dataset

The dataset used consists of labeled news articles categorized as *fake* or *real*. The labels are based on the credibility of the sources. The textual content of the articles is analyzed to classify them using machine learning algorithms.

- **Dataset Information**: Text-based news articles with labels.
- **Preprocessing**: Tokenization, stopwords removal, and lemmatization are applied to clean and process the text.
- **Text Features**: The text is transformed into numerical representations using TF-IDF (Term Frequency-Inverse Document Frequency).

## Models

We implemented and evaluated multiple machine learning models for fake news detection. The models include:
- **Logistic Regression**: A linear model for binary classification.
- **Random Forest Classifier**: An ensemble method with multiple decision trees.

### Preprocessing and Model Training
- **spaCy**: Used for text preprocessing.
- **scikit-learn**: Utilized for building machine learning models and evaluating their performance.
- **TF-IDF**: Used for vectorizing text to numerical features.

All models were evaluated on the dataset, and their performance metrics such as accuracy, precision, recall, and F1-score were recorded.

## Environment

The project was developed and tested using the following environment:
- **Python**: 3.8+
- **spaCy**: 3.x
- **scikit-learn**: 0.24.2
- **joblib**: 1.0.1 (for model saving/loading)
- **Streamlit**: 0.85.0 (for the web app interface)

Please refer to the `requirements.txt` file for a complete list of dependencies.

### Setup Instructions

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/fake-news-detection.git
    cd fake-news-detection
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Download necessary NLP models (spaCy):
    ```bash
    python -m spacy download en_core_web_sm
    ```

### Application for Data Use
If you plan to use the dataset for research or production purposes, please ensure it complies with your institution's policies on data privacy and ethics.

## Data Processing

- **spaCy**: Used for text tokenization, stopwords removal, and lemmatization.
- **TF-IDF Vectorizer**: Converts the text into numerical vectors for machine learning models.
- **Models**: Logistic Regression and Random Forest Classifier were used for prediction. The models can be loaded using **joblib**.

You can either extract features using your own preprocessing pipeline or use the pre-trained models provided in this repository.

### Pre-Trained Models
The repository includes pre-trained models for quick inference. These models have been trained and saved as `.pkl` files for ease of use. To load a model and make predictions, refer to the `predict.py` script.

## Model Benchmarks

The performance of the models was evaluated on the dataset with the following benchmarks:
- **Accuracy**: Logistic Regression: 90%, Random Forest: 92%
- **Precision**: Logistic Regression: 89%, Random Forest: 91%
- **Recall**: Logistic Regression: 88%, Random Forest: 90%
- **F1-Score**: Logistic Regression: 88.5%, Random Forest: 90.5%

## Streamlit Web Application

A web-based user interface is provided through **Streamlit**. You can run the application by executing the following command:

```bash
streamlit run app.py
```

The app allows users to input news articles, and the system will classify them as *real* or *fake* based on the trained models.

---

## Future Improvements

- **Multimodal Analysis**: Extend the system to handle videos and images, integrating text analysis with visual cues.
- **Advanced NLP Models**: Incorporate state-of-the-art language models such as BERT for better accuracy.
- **Cross-Domain Testing**: Apply the system to various forms of text, such as social media posts and blogs.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## Contact

For questions or inquiries, please reach out to:

- **Muhammad Waqar Ali**
- Email: [itxmewaqar@gmail.com](mailto:itxmewaqar@gmail.com)
