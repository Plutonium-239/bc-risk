import pandas as pd
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler

toplot = ['IES-R Score', 'FinalDass21Score', 'Distress', 'Anxiety', 'Stress', 'any PTSD yesorno']
scaler = MinMaxScaler()
df = pd.read_excel('../data/Preprocessed_dataset v1.xlsx', engine='openpyxl')
norm_df = df[toplot].values
norm_df = scaler.fit_transform(norm_df)
norm_df = pd.DataFrame(norm_df)
norm_df.columns = toplot
norm_df['Sno'] = df['Sno']

px.scatter(norm_df, x='Sno', y=toplot).show()