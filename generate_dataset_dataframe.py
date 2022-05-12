import pandas as pd
from pathlib import Path

used_dataset = 'dataset/default'

yesDir = f'{used_dataset}/train/yes'
paths = []
for ext in ['*.jpg', '*.jpeg', '*.JPG', '*.png']:
    paths = [*paths, *[path.parts[-3:] for path in Path(used_dataset).rglob(ext)]]

df = pd.DataFrame(data=paths, columns=['Dataset', 'State', 'Path'])
df.to_csv('dataframe.csv')
