import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score


class RENTPrediction:
    def __init__(self, file_path):
        data = joblib.load(file_path)
        self.random_model = data['random_model']
        self.one_hot_encoder = data['one_hot_encoder']
        self.ordinal_encoder = data['ordinal_encoder']
        self.standard_scaler = data['standard_scaler']
        self.mean = data['mean']
        self.modes = data['modes']
        self.columns = data['columns']

    def pipeline(self, in_comming_data):
        df = in_comming_data.copy()

        numeric_cols = df.select_dtypes(include=["number"]).columns
        category_cols = df.select_dtypes(include=["object"]).columns

        print(f"Numeric cols : {numeric_cols}")
        print(f"Category cols : {category_cols}")

        for col, val in self.mean.items():
            if col in list(numeric_cols):
                df[col] = df[col].fillna(val)
        
        for col, val in self.modes.items():
            if col in list(category_cols):
                df[col] = df[col].fillna(val)

        cols_to_encode = [
            "Satisfaction_Salaire",
            "Equilibre_Vie_Travail",
        ]

        existing_encode_cols = [col for col in cols_to_encode if col in df.columns]

        if existing_encode_cols and self.ordinal_encoder:
            df[existing_encode_cols] = self.ordinal_encoder.transform(df[existing_encode_cols])

        missing_cols = [c for c in self.columns if c not in df.columns]

        if missing_cols:
            print(f"Missing cols : {missing_cols}")
            raise ValueError(f"Columns missed --> {missing_cols}")
        
        cols_to_scale = [
            "Age",
            "Salaire_Mensuel_BIF",
            "Heures_Supplementaires",
            "Heures_Formation",
            "Nombre_Absences"
        ]

        print(df.columns)
        print()
        print(self.columns)

        cols = [col for col in cols_to_scale if col in df.columns]

        if cols and self.standard_scaler:
            df[cols] = self.standard_scaler.transform(df[cols])

        try:
            X_final = df[self.columns]
        except KeyError as e:
            raise ValueError(f"Columns doesn't match : {e}")

        return X_final

    def predict(self, in_comming_data):
        X = self.pipeline(in_comming_data)

        prediction_value = self.random_model.predict(X)

        print(f"Prediction (log scale): {prediction_value}")
        
        return prediction_value
