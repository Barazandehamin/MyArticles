import pandas as pd

data= pd.read_csv("source_price.csv")

print(data.shape)
print(data.head)
#print(data.describe())
data['reuters_mean_compound'].plot()
data['wsj_mean_compound'].plot()
data['cnbc_mean_compound'].plot()
data['fortune_mean_compound'].plot()

