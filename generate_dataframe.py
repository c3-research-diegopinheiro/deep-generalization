import pandas as pd
from dataset_path import dataset_path
from pathlib import Path

yesDir = dataset_path + '/train/yes'
paths = [path.parts[-3:] for path in Path(dataset_path).rglob('*.jpg')]
paths2 = [path.parts[-3:] for path in Path(dataset_path).rglob('*.jpeg')]
paths3 = [path.parts[-3:] for path in Path(dataset_path).rglob('*.JPG')]
paths4 = [path.parts[-3:] for path in Path(dataset_path).rglob('*.png')]

df = pd.DataFrame(data=[*paths, *paths2, *paths3, *paths4], columns=['Dataset', 'State', 'Path'])
df.to_csv('dataframe.csv')

pie = df.groupby(['Dataset']).count().plot(kind='pie', y='State', autopct='%1.0f%%')

fig = pie.get_figure()
fig.savefig("plots/dataset-pie.png")
