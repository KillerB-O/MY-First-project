import pandas as pd
from house_predicting_model import make_prediction
import seaborn as sns
import matplotlib.pyplot as plt

data=pd.read_csv('house price data.csv')
top_10_streets=data.groupby('street')["price"].mean()
top_10_streets=top_10_streets.sort_values(ascending=False).head(10)
plt.figure(figsize = (10,8))
sns.barplot(y=top_10_streets.index, x=top_10_streets.values)
plt.title('Top 10 streets price')
plt.xlabel('mean price')
plt.ylabel('street name')
plt.show()

bathroom=int(input("Enter the no of bathrooms: "))
predicted_price=make_prediction(data,bathroom)
print(f"The predicted price for {bathroom} bathroom house is : {predicted_price:,.2f}")


