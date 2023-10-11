from tensorflow.keras.layers import Flatten, Dense

from nid import Nid

dataframe_path = f'./dataset/dataframe.csv'

nid = Nid([Flatten(input_shape=(200, 200, 3)),
           Dense(4, activation='relu'),
           Dense(1, activation='sigmoid')],
          (5, 5, 3),
          1,
          1e-3,
          2,
          dataframe_path,
          )

nid.execute()
