# News Article Classification Project 📰

![Static Badge](https://img.shields.io/badge/Status-Complete-green)

## Description

This project aims to develop a machine learning model capable of classifying news articles into different categories based on their titles.  Two different word embedding models (CBOW and Skip-gram) are trained and used to vectorize the article titles. These vectorized representations are then used to train a Logistic Regression classifier.

## Technologies Used

- Python
- Pandas
- Scikit-learn (Logistic Regression, `classification_report`)
- spaCy (Portuguese language model)
- Gensim (Word2Vec, KeyedVectors)
- Joblib (for model persistence)
- Requests (for downloading data)
- Pathlib (for file path management)


## Detailed Project Description

This project follows these key steps:

1. **Data Acquisition:**  The training and testing datasets are downloaded from specified URLs and cached locally as CSV files using the `download_csv` function.  This function utilizes the `requests` library for robust downloads and caching with `lru_cache` for efficiency.

2. **Text Preprocessing:** Article titles are cleaned and tokenized using spaCy's Portuguese language model (`pt_core_news_sm`). The `get_text_from_valid_tokens` and `tokenizer` functions handle removing stop words, punctuation, and converting text to lowercase.  This prepares the text data for use in the word embedding models.

3. **Word Embedding Model Training:**
    - **CBOW Model:** A Continuous Bag-of-Words (CBOW) model is trained using Gensim's `Word2Vec` implementation. Parameters such as vector size, window size, minimum word count, learning rate (`alpha`), and minimum learning rate (`min_alpha`) are specified. The model learns vector representations of words based on their surrounding context.
    - **Skip-gram Model:** A Skip-gram model is also trained using `Word2Vec`, learning word vectors by predicting surrounding words given a target word.  Similar parameters to the CBOW model are configured.

4. **Vectorization:** The trained word embedding models are used to convert the preprocessed article titles into numerical vectors.  The `get_vectors` function tokenizes each title and uses the `sum_tokens_vector` function to create a single vector representation for each title by summing the vectors of the individual words.

5. **Classification Model Training:** A Logistic Regression model is trained using Scikit-learn's `LogisticRegression`. The vectorized titles from the training set serve as input features, and the corresponding article categories are the target variable. The model is trained with a specified maximum number of iterations (`max_iter`).

6. **Model Evaluation:** The trained Logistic Regression models (one for each embedding method) are evaluated on the test set using Scikit-learn's `classification_report`. This report provides key metrics such as precision, recall, F1-score, and support for each category, as well as overall accuracy, macro average, and weighted average.

7. **Model Persistence:**  Both the trained word embedding models (in text format) and the Logistic Regression models (using Joblib) are saved to disk for later use. This allows for loading the models without retraining.


## Data Dictionary

| Column | Description | Data Type |
|---|---|---|
| title | The title of the news article. | String |
| text | The full text content of the news article. | String |
| date | The publication date of the article. | Date |
| category | The category of the news article (target variable). | String |
| subcategory | A more specific classification of the article (not used in this project). | String |
| link | The URL of the news article. | String |

## Files

- `data/treino.csv`: Training dataset.
- `data/teste.csv`: Test dataset.
- `models/model_cbow.txt`: Saved CBOW word embedding model.
- `models/model_skipgram.txt`: Saved Skip-gram word embedding model.
- `models/cbow_lr_model.joblib`: Saved Logistic Regression model trained with CBOW embeddings.
- `models/skipgram_lr_model.joblib`: Saved Logistic Regression model trained with Skip-gram embeddings.
