import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler


class PreprocessingDataset:
    def __init__(self, x_train: pd.DataFrame, list_var: list) -> None:
        self.x_train = x_train
        self.list_var = list_var

    def preprocessing_dataset(self):
        cat_cols = self.x_train.select_dtypes(
            include=["object", "category"]
        ).columns.to_list()
        numeric_cols = self.x_train.select_dtypes(
            include=["int64", "float64"]
        ).columns.to_list()

        # para eliminar una de las dummies cuando se esta aplicando regresión lineal o logistica se usa drop='first' en onehoten
        preprocessor = ColumnTransformer(
            [
                ("scale", StandardScaler(), numeric_cols),
                ("onehot", OneHotEncoder(drop="first"), cat_cols),
            ],
            remainder="passthrough",
        )

        if len(self.list_var) >= 5:
            X_train_prep = preprocessor.fit_transform(self.x_train)
            encoded_cat = preprocessor.named_transformers_[
                "onehot"
            ].get_feature_names_out(cat_cols)
            labels = np.concatenate([numeric_cols, encoded_cat])
            X_train_prep_matrix = X_train_prep.toarray()
            datos_train_prep = pd.DataFrame(X_train_prep_matrix, columns=labels)
            return datos_train_prep

        X_train_prep = preprocessor.fit_transform(self.x_train)
        encoded_cat = preprocessor.named_transformers_["onehot"].get_feature_names_out(
            cat_cols
        )
        labels = np.concatenate([numeric_cols, encoded_cat])
        datos_train_prep = pd.DataFrame(X_train_prep, columns=labels)

        return datos_train_prep
