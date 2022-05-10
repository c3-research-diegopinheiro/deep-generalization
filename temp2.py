import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('output/cnn_results_2022-04-27T01:24:15.999932/cnn_results.csv')

# data = df[['train-dataset-noise', 'test-dataset-noise', 'f1-score(weighted-avg)']].to_numpy()

sns.set_theme(style="whitegrid")

g = sns.catplot(
    data=df, kind="bar",
    x="train-dataset-noise", y="f1-score(weighted-avg)", hue="test-dataset-noise",
    ci="sd", palette="dark", alpha=.6, height=6
)
g.despine(left=True)
g.set(xlabel='common xlabel', ylabel='common ylabel')
g.legend.set_title("")
