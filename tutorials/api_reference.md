# phoenix.Dataset

```python
class Dataset(
    dataframe: pandas.DataFrame,
    schema: Schema,
    name: Optional[str] = None,
)
```

A dataset containing a split or cohort of data to be analyzed independently or compared to another cohort. Common examples include training, validation, test, or production datasets.

## Parameters

* **dataframe** (pandas.DataFrame): The data to be analyzed or compared.
* **schema** (phoenix.Schema): A schema that assigns the columns of the dataframe to the appropriate model dimensions (features, predictions, actuals, etc.).
* **name** (Optional[str]): The name used to identify the dataset in the application. If not provided, a random name will be generated.

## Methods

* **head**(num_rows: Optional[int] = 5) -> pandas.DataFrame

  Returns the first `num_rows` rows of the dataset's dataframe. This method is useful for inspecting the dataset's underlying dataframe to ensure it has the expected format and content.

  **Parameters**
  * **num_rows** (int): The number of rows in the returned dataframe.

## Attributes

* **dataframe** (pandas.DataFrame): The pandas dataframe of the dataset.
* **schema** (phoenix.Schema): The schema of the dataset.
* **name** (str): The name of the dataset.

## Usage

Define a dataset `ds` from a pandas dataframe `df` and a schema object `schema` by running

```python
ds = px.Dataset(df, schema)
```

Alternatively, provide a name for the dataset that will appear in the application:

```python
ds = px.Dataset(df, schema, name="training")
```

`ds` is then passed as the `primary` or `reference` argument to [phoenix.launch_app](session.md#phoenix.launch_app).

# phoenix.Schema

```python
class Schema(
    prediction_id_column_name: Optional[str] = None,
    timestamp_column_name: Optional[str] = None,
    feature_column_names: Optional[List[str]] = None,
    tag_column_names: Optional[List[str]] = None,
    prediction_label_column_name: Optional[str] = None,
    prediction_score_column_name: Optional[str] = None,
    actual_label_column_name: Optional[str] = None,
    actual_score_column_name: Optional[str] = None,
    prompt_column_names: Optional[EmbeddingColumnNames] = None
    response_column_names: Optional[EmbeddingColumnNames] = None
    embedding_feature_column_names: Optional[Dict[str, EmbeddingColumnNames]] = None,
    excluded_column_names: Optional[List[str]] = None,
)
```

A dataclass that assigns the columns of a pandas dataframe to the appropriate model dimensions (predictions, actuals, features, etc.). Each column of the dataframe should appear in the corresponding schema at most once.

## Parameters

* **prediction_id_column_name** (Optional[str]): The name of the dataframe's prediction ID column, if one exists. Prediction IDs are strings that uniquely identify each record in a Phoenix dataset (equivalently, each row in the dataframe). If no prediction ID column name is provided, Phoenix will automatically generate unique UUIDs for each record of the dataset upon phoenix.Dataset initialization.
* **timestamp_column_name** (Optional[str]): The name of the dataframe's timestamp column, if one exists. Timestamp columns must be pandas Series with numeric or datetime dtypes.
  * If the timestamp column has numeric dtype (`int` or `float`), the entries of the column are interpreted as Unix timestamps, i.e., the number of seconds since midnight on January 1st, 1970.
  * If the column has datetime dtype and contains timezone-naive timestamps, Phoenix assumes those timestamps belong to the UTC timezone.
  * If the column has datetime dtype and contains timezone-aware timestamps, those timestamps are converted to UTC.
  * If no timestamp column is provided, each record in the dataset is assigned the current timestamp upon phoenix.Dataset initialization.
* **feature_column_names** (Optional[List[str]]): The names of the dataframe's feature columns, if any exist. If no feature column names are provided, all dataframe column names that are not included elsewhere in the schema and are not explicitly excluded in `excluded_column_names` are assumed to be features.
* **tag_column_names** (Optional[List[str]]): The names of the dataframe's tag columns, if any exist. Tags, like features, are attributes that can be used for filtering records of the dataset while using the app. Unlike features, tags are not model inputs and are not used for computing metrics.
* **prediction_label_column_name** (Optional[str]): The name of the dataframe's predicted label column, if one exists. Predicted labels are used for classification problems with categorical model output.
* **prediction_score_column_name** (Optional[str]): The name of the dataframe's predicted score column, if one exists. Predicted scores are used for regression problems with continuous numerical model output.
* **actual_label_column_name** (Optional[str]): The name of the dataframe's actual label column, if one exists. Actual (i.e., ground truth) labels are used for classification problems with categorical model output.
* **actual_score_column_name** (Optional[str]): The name of the dataframe's actual score column, if one exists. Actual (i.e., ground truth) scores are used for regression problems with continuous numerical output.
* **prompt_column_names** (Optional[phoenix.EmbeddingColumnNames]): An instance of phoenix.EmbeddingColumnNames delineating the column names of an LLM model's prompt embedding vector, prompt text, and optionally links to external resources.
* **response_column_names** (Optional[phoenix.EmbeddingColumnNames]): An instance of phoenix.EmbeddingColumnNames delineating the column names of an LLM model's _response_ embedding vector, _response_ text, and optionally links to external resources.
* **embedding_feature_column_names** (Optional[Dict[str, phoenix.EmbeddingColumnNames]]): A dictionary mapping the name of each embedding feature to an instance of phoenix.EmbeddingColumnNames if any embedding features exist, otherwise, None. Each instance of phoenix.EmbeddingColumnNames associates one or more dataframe columns containing vector data, image links, or text with the same embedding feature. Note that the keys of the dictionary are user-specified names that appear in the Phoenix UI and do not refer to columns of the dataframe.
* **excluded_column_names** (Optional[List[str]]): The names of the dataframe columns to be excluded from the implicitly inferred list of feature column names. This field should only be used for implicit feature discovery, i.e., when `feature_column_names` is unused and the dataframe contains feature columns not explicitly included in the schema.

# phoenix.EmbeddingColumnNames

```python
class EmbeddingColumnNames(
    vector_column_name: str,
    raw_data_column_name: Optional[str] = None,
    link_to_data_column_name: Optional[str] = None,
)
```

A dataclass that associates one or more columns of a dataframe with an embedding feature. Instances of this class are only used as values in a dictionary passed to the `embedding_feature_column_names` field of phoenix.Schema.

## Parameters

* **vector_column_name** (str): The name of the dataframe column containing the embedding vector data. Each entry in the column must be a list, one-dimensional NumPy array, or pandas Series containing numeric values (floats or ints) and must have equal length to all the other entries in the column.
* **raw_data_column_name** (Optional[str]): The name of the dataframe column containing the raw text associated with an embedding feature, if such a column exists. This field is used when an embedding feature describes a piece of text, for example, in the context of NLP.
* **link_to_data_column_name** (Optional[str]): The name of the dataframe column containing links to images associated with an embedding feature, if such a column exists. This field is used when an embedding feature describes an image, for example, in the context of computer vision.