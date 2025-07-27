import pandas as pd
from sklearn.linear_model import LinearRegression
def make_prediction(data,bathroom):
    X=data[["bathrooms"]]
    y=data["price"]

    model = LinearRegression()

    model.fit(X,y)

    new_house_data=pd.DataFrame({"bathrooms":[bathroom]})

    prediction=model.predict(new_house_data)
    return prediction[0]

