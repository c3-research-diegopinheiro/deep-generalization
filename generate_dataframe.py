import pandas as pd
from pathlib import Path

used_dataset = 'DATASET'

yesDir = used_dataset + '/train/yes'
paths = []
for ext in ['*.jpg', '*.jpeg', '*.JPG', '*.png']:
    paths = [*paths, *[path.parts[-3:] for path in Path(used_dataset).rglob(ext)]]

df = pd.DataFrame(data=paths, columns=['Dataset', 'State', 'Path'])
df.to_csv('dataframe.csv')

pie = df.groupby(['Dataset']).count().plot(kind='pie', y='State', autopct='%1.0f%%')

fig = pie.get_figure()
fig.savefig("plots/dataset-division-from-" + used_dataset + ".png")
