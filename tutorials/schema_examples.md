# Binary Classification

The example dataframe below contains inference data from a binary classification model trained to predict whether a user will click on an advertisement. The timestamps are `datetime.datetime` objects that represent the time at which each inference was made in production.

## Dataframe

```python
import pandas as pd

df = pd.DataFrame([
    [pd.to_datetime('2023-03-01 02:02:19'), 0.91, 'click', 'click'],
    [pd.to_datetime('2023-02-17 23:45:48'), 0.37, 'no_click', 'no_click'],
    [pd.to_datetime('2023-01-30 15:30:03'), 0.54, 'click', 'no_click'],
    [pd.to_datetime('2023-02-03 19:56:09'), 0.74, 'click', 'click'],
    [pd.to_datetime('2023-02-24 04:23:43'), 0.37, 'no_click', 'click']
], columns=['timestamp', 'prediction_score', 'prediction', 'target'])
```

## Schema

```python
schema = px.Schema(
    timestamp_column_name="timestamp",
    prediction_score_column_name="prediction_score",
    prediction_label_column_name="prediction",
    actual_label_column_name="target",
)
```

# Features and Tags

Phoenix accepts not only predictions and ground truth but also input features of your model and tags that describe your data. In the example below, features such as FICO score and merchant ID are used to predict whether a credit card transaction is legitimate or fraudulent. In contrast, tags such as age and gender are not model inputs, but are used to filter your data and analyze meaningful cohorts in the app.

## Dataframe

```python
df = pd.DataFrame({
    'fico_score': [578, 507, 656, 414, 512],
    'merchant_id': ['Scammeds', 'Schiller Ltd', 'Kirlin and Sons', 'Scammeds', 'Champlin and Sons'],
    'loan_amount': [4300, 21000, 18000, 18000, 20000],
    'annual_income': [62966, 52335, 94995, 32034, 46005],
    'home_ownership': ['RENT', 'RENT', 'MORTGAGE', 'LEASE', 'OWN'],
    'num_credit_lines': [110, 129, 31, 81, 148],
    'inquests_in_last_6_months': [0, 0, 0, 2, 1],
    'months_since_last_delinquency': [0, 23, 0, 0, 0],
    'age': [25, 78, 54, 34, 49],
    'gender': ['male', 'female', 'female', 'male', 'male'],
    'predicted': ['not_fraud', 'not_fraud', 'uncertain', 'fraud', 'uncertain'],
    'target': ['fraud', 'not_fraud', 'uncertain', 'not_fraud', 'uncertain']
})
```

## Schema

```python
schema = px.Schema(
    prediction_label_column_name="predicted",
    actual_label_column_name="target",
    feature_column_names=[
        "fico_score",
        "merchant_id",
        "loan_amount",
        "annual_income",
        "home_ownership",
        "num_credit_lines",
        "inquests_in_last_6_months",
        "months_since_last_delinquency",
    ],
    tag_column_names=[
        "age",
        "gender",
    ],
)
```

# Implicit Features

If your data has a large number of features, it can be inconvenient to list them all. For example, the breast cancer dataset below contains 30 features that can be used to predict whether a breast mass is malignant or benign. Instead of explicitly listing each feature, you can leave the `feature_column_names` field of your schema set to its default value of `None`, in which case, any columns of your dataframe that do not appear in your schema are implicitly assumed to be features.

## Dataframe

```python
df = pd.DataFrame([
    {
        'target': 'malignant',
        'predicted': 'benign',
        'mean radius': 15.49,
        'mean texture': 19.97,
        'mean perimeter': 102.40,
        'mean area': 744.7,
        'mean smoothness': 0.11600,
        'mean compactness': 0.15620,
        'mean concavity': 0.18910,
        'mean concave points': 0.09113,
        'mean symmetry': 0.1929,
        'mean fractal dimension': 0.06744,
        'radius error': 0.6470,
        'texture error': 1.3310,
        'perimeter error': 4.675,
        'area error': 66.91,
        'smoothness error': 0.007269,
        'compactness error': 0.02928,
        'concavity error': 0.04972,
        'concave points error': 0.01639,
        'symmetry error': 0.01852,
        'fractal dimension error': 0.004232,
        'worst radius': 21.20,
        'worst texture': 29.41,
        'worst perimeter': 142.10,
        'worst area': 1359.0,
        'worst smoothness': 0.1681,
        'worst compactness': 0.3913,
        'worst concavity': 0.55530,
        'worst concave points': 0.21210,
        'worst symmetry': 0.3187,
        'worst fractal dimension': 0.10190
    },
])
```

## Schema

```python
schema = px.Schema(
    prediction_label_column_name="predicted",
    actual_label_column_name="target",
)
```

# Only Embedding Vectors

To define an embedding feature, you must at minimum provide Phoenix with the embedding vector data itself. Specify the dataframe column that contains this data in the `vector_column_name` field on `px.EmbeddingColumnNames`. For example, the dataframe below contains tabular credit card transaction data in addition to embedding vectors that represent each row. Notice that:

* Unlike other fields that take strings or lists of strings, the argument to `embedding_feature_column_names` is a dictionary.
* The key of this dictionary, "transaction_embedding," is not a column of your dataframe but is name you choose for your embedding feature that appears in the UI.
* The values of this dictionary are instances of `px.EmbeddingColumnNames`.
* Each entry in the "embedding_vector" column is a list of length 4.

## Dataframe

```python
df = pd.DataFrame({
    'predicted': ['fraud', 'fraud', 'not_fraud', 'not_fraud', 'uncertain'],
    'target': ['not_fraud', 'not_fraud', 'not_fraud', 'not_fraud', 'uncertain'],
    'embedding_vector': [[-0.97, 3.98, -0.03, 2.92], [3.20, 3.95, 2.81, -0.09], [-0.49, -0.62, 0.08, 2.03], [1.69, 0.01, -0.76, 3.64], [1.46, 0.69, 3.26, -0.17]],
    'fico_score': [604, 612, 646, 560, 636],
    'merchant_id': ['Leannon Ward', 'Scammeds', 'Leannon Ward', 'Kirlin and Sons', 'Champlin and Sons'],
    'loan_amount': [22000, 7500, 32000, 19000, 10000],
    'annual_income': [100781, 116184, 73666, 38589, 100251],
    'home_ownership': ['RENT', 'MORTGAGE', 'RENT', 'MORTGAGE', 'MORTGAGE'],
    'num_credit_lines': [108, 42, 131, 131, 10],
    'inquests_in_last_6_months': [0, 2, 0, 0, 0],
    'months_since_last_delinquency': [0, 56, 0, 0, 3]
})
```

## Schema

```python
schema = px.Schema(
    prediction_label_column_name="predicted",
    actual_label_column_name="target",
    embedding_feature_column_names={
        "transaction_embeddings": px.EmbeddingColumnNames(
            vector_column_name="embedding_vector"
        ),
    },
)
```

# Embeddings of Images

If your embeddings represent images, you can provide links or local paths to image files you want to display in the app by using the `link_to_data_column_name` field on `px.EmbeddingColumnNames`. The following example contains data for an image classification model that detects product defects on an assembly line.

## Dataframe

```python
df = pd.DataFrame({
    'defective': ['okay', 'defective', 'okay', 'defective', 'okay'],
    'image': ['https://www.example.com/image0.jpeg', 'https://www.example.com/image1.jpeg', 'https://www.example.com/image2.jpeg', 'https://www.example.com/image3.jpeg', 'https://www.example.com/image4.jpeg'],
    'image_vector': [[1.73, 2.67, 2.91, 1.79, 1.29], [2.18, -0.21, 0.87, 3.84, -0.97], [3.36, -0.62, 2.40, -0.94, 3.69], [2.77, 2.79, 3.36, 0.60, 3.10], [1.79, 2.06, 0.53, 3.58, 0.24]]
})
```

## Schema

```python
schema = px.Schema(
    actual_label_column_name="defective",
    embedding_feature_column_names={
        "image_embedding": px.EmbeddingColumnNames(
            vector_column_name="image_vector",
            link_to_data_column_name="image",
        ),
    },
)
```

# Embeddings of Text

If your embeddings represent pieces of text, you can display that text in the app by using the `raw_data_column_name` field on `px.EmbeddingColumnNames`. The embeddings below were generated by a sentiment classification model trained on product reviews.

## Dataframe

```python
df = pd.DataFrame({
    'defective': ['okay', 'defective', 'okay', 'defective', 'okay'],
    'image': ['https://www.example.com/image0.jpeg', 'https://www.example.com/image1.jpeg', 'https://www.example.com/image2.jpeg', 'https://www.example.com/image3.jpeg', 'https://www.example.com/image4.jpeg'],
    'image_vector': [[1.73, 2.67, 2.91, 1.79, 1.29], [2.18, -0.21, 0.87, 3.84, -0.97], [3.36, -0.62, 2.40, -0.94, 3.69], [2.77, 2.79, 3.36, 0.60, 3.10], [1.79, 2.06, 0.53, 3.58, 0.24]]
})
```

## Schema

```python
schema = px.Schema(
    actual_label_column_name="sentiment",
    feature_column_names=[
        "category",
    ],
    tag_column_names=[
        "name",
    ],
    embedding_feature_column_names={
        "product_review_embeddings": px.EmbeddingColumnNames(
            vector_column_name="text_vector",
            raw_data_column_name="text",
        ),
    },
)
```
