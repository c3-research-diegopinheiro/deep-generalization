import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df_simple_mlp = pd.read_csv('output/results_simple-mlp_2022-04-27T23:48:29.134349/results_simple-mlp.csv')
df_cnn = pd.read_csv('output/cnn_results_2022-04-27T01:24:15.999932/cnn_results.csv')

data = []
block = []
arr_simple_mlp = np.array(df_simple_mlp[['f1-score(weighted-avg)']])
arr_cnn = np.array(df_cnn[['f1-score(weighted-avg)']])

arr_simple_mlp_2d = np.reshape(arr_simple_mlp, (10, 10))
arr_cnn_2d = np.reshape(arr_cnn, (10, 10))

variance_simple_mlp = []
variance_cnn = []
for x in arr_simple_mlp_2d:
    x.sort()
    variance_simple_mlp.append(((x[len(x) - 1] - x[0]) / x[0]) * 100)
for y in arr_cnn_2d:
    y.sort()
    variance_cnn.append(((y[len(y) - 1] - y[0]) / y[0]) * 100)

print('simple_mlp')
print(variance_simple_mlp)
print('cnn')
print(variance_cnn)
d = {
    'Train Noise': [0, .1, .2, .3, .4, .5, .6, .7, .8, .9],
    'Simple MLP': variance_simple_mlp,
    'CNN': variance_cnn,
}
df = pd.DataFrame(data=d)

print(df)




# sns.set_theme(style="whitegrid")
#
# g = sns.catplot(
#     data=df, kind="bar",
#     x="train-dataset-noise", y="f1-score(weighted-avg)", hue="test-dataset-noise",
#     ci="sd", palette="dark", alpha=.6, height=6
# )
# g.despine(left=True)
# g.set(xlabel='common xlabel', ylabel='common ylabel')
# g.legend.set_title("")
