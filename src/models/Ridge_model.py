import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.pipeline import make_pipeline

class Ridge_model:
    def __init__(self, city):
        self.csv_file = f"prices/House_Prices_{city}.csv"
        self.y_columns=["x","y","m2","house_type","standard"]
        self.pipeline = make_pipeline(PolynomialFeatures(degree=3),
                                      StandardScaler(),
                                      Ridge(alpha=.3),)

        self.fitted = False

    def build_and_train_model(self):
        df:DataFrame = pd.read_csv(self.csv_file)

        # --- Load your data ---
        y = df["sales_price"].values
        x = df[self.y_columns]
        x["x2"] = x["x"] ** 1.5
        x["y2"] = x["y"] ** 1.5

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

        self.pipeline.fit(x_train, y_train)
        self.fitted = True
        y_pred = self.pipeline.predict(x_test).astype(int)

        # Check model quality
        l_test:list[int] = y_test.tolist()
        l_pred:list[int] = y_pred.tolist()
        for i in range(0, min(len(l_test),10)):
            print(f"Price: {l_test[i]:>10,.0f}  Predicted: {l_pred[i]:>10,.0f}")
        print(f"R-squared: {r2_score(y_test, y_pred):.2f}")
        print(f"MAE      : {mean_absolute_error(y_test, y_pred):.0f}")

    def predict_price(self, x, y, m2:int,house_type, standard) -> float:
        if not self.fitted:
            raise RuntimeError("Model is not fitted yet")
        # Build a single sample
        x_house = pd.DataFrame([[x, y,m2,house_type,standard]],columns=self.y_columns)
        x_house['x2'] = x_house['x']**1.5 # Weith position more.
        x_house['y2'] = x_house['y']**1.5
        pred = self.pipeline.predict(x_house)[0]
        return float(pred)



