import logging
import pickle
import sys

import pandas as pd
from category_encoders import OrdinalEncoder
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def wrangle(filename):
    """
    Loads and preprocesses the dataset.

    Args:
        filename (str): The path to the input CSV file.

    Returns:
        pd.DataFrame: The processed DataFrame.
    """
    try:
        df = pd.read_csv(filename)
        # df = df[df["Season"] != "2017/18"]
        return df
    except ValueError as e:
        logging.error("Error loading file: %s", e)
        sys.exit(1)


def preprocess_data(df):
    """
    Preprocesses the DataFrame for model training.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        tuple: Processed feature matrix and target vector.
    """
    X = df[["home", "away", "home_form", "away_form"]]
    y = df["winner"].map({"home": 1, "away": -1, "draw": 0})
    return X, y


def build_pipeline():
    """
    Builds a machine learning pipeline with ordinal encoding and a random forest classifier.

    Returns:
        Pipeline: The constructed machine learning pipeline.
    """
    pipeline = make_pipeline(
        OrdinalEncoder(),
        SimpleImputer(fill_value=0),
        RandomForestClassifier(
            n_estimators=250,
            min_samples_split=150,
            min_impurity_decrease=0.0001,
            random_state=42,
            n_jobs=-1,
            max_features="sqrt",
        ),
    )
    return pipeline


def evaluate_model(model, X_train, y_train, X_test, y_test):
    """
    Evaluates the model on the training and test sets.

    Args:
        model (Pipeline): The trained model.
        X_train (pd.DataFrame): Training feature matrix.
        y_train (pd.Series): Training target vector.
        X_test (pd.DataFrame): Test feature matrix.
        y_test (pd.Series): Test target vector.
    """
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)
    logging.info("Baseline Accuracy: %s", dummy.score(X_test, y_test))

    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    logging.info("Training Accuracy: %s", train_acc)
    logging.info("Test Accuracy: %s", test_acc)

    y_pred = model.predict(X_test)
    logging.info("Confusion Matrix:")
    logging.info("\n%s", confusion_matrix(y_test, y_pred))
    logging.info("Classification Report:")
    logging.info(classification_report(y_test, y_pred))


def save_model(model, filepath):
    """
    Saves the trained model to a file.

    Args:
        model (Pipeline): The trained model.
        filepath (str): The path to the output file.
    """
    with open(f"{filepath}.pkl", "wb") as f:
        pickle.dump(model, f)


def load_model(filepath):
    """
    Load the trained model from a file.

    Args:
        filepath (str): The path to the trained model file.

    Returns:
        Pipeline: The trained model.
    """
    with open(f"{filepath}.pkl", "rb") as f:
        model = pickle.load(f)
    return model


def predict(model, X):
    """
    Predicts the outcome of a match.

    Args:
        model (Pipeline): The trained model.
        X (pd.DataFrame): The input features.

    Returns:
        pd.Series: The predicted outcomes.
    """
    predictions = model.predict(X)
    return pd.Series(predictions).map({1: "home", -1: "away", 0: "draw"})


def predict_match(home_team, away_team, home_form, away_form, model):
    """
    Predicts the outcome of a match.

    Args:
        home_team (str): The home team name.
        away_team (str): The away team name.
        home_form (int): The home team's form.
        away_form (int): The away team's form.
        model (Pipeline): The trained model.

    Returns:
        str: The predicted outcome.
    """
    X = pd.DataFrame(
        {
            "home": [home_team],
            "away": [away_team],
            "home_form": [home_form],
            "away_form": [away_form],
        }
    )
    prediction = predict(model, X)
    return prediction[0]


def predict_proba(model, X):
    """
    Predicts the probability of each outcome of a match.

    Args:
        model (Pipeline): The trained model.
        X (pd.DataFrame): The input features.

    Returns:
        pd.DataFrame: The predicted probabilities for each outcome.
    """
    probabilities = model.predict_proba(X)
    return pd.DataFrame(
        {
            "home": probabilities[:, 0],
            "draw": probabilities[:, 1],
            "away": probabilities[:, 2],
        }
    )


def main(csv_file, out_model_filepath):
    """
    Main function to load data, train the model, evaluate it, and save the model.

    Args:
        csv_file (str): The path to the input CSV file.
        model_filepath (str): The path to save the trained model.
    """
    df = wrangle(csv_file)
    X, y = preprocess_data(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = build_pipeline()
    model.fit(X_train, y_train)
    evaluate_model(model, X_train, y_train, X_test, y_test)
    save_model(model, out_model_filepath)


if __name__ == "__main__":

    if len(sys.argv) != 3:
        print("Usage: python script.py <input_csv> <model_filepath>")
        sys.exit(1)

    input_csv = sys.argv[1]
    model_filepath = sys.argv[2]
    main(input_csv, model_filepath)
