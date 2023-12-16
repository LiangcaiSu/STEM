# %%
import numpy as np 
import pandas as pd 
# from datetime import datetime
from sklearn.model_selection import train_test_split

# %%
data = pd.read_csv('./raw_data/final_track1_train.txt',sep='\t',
                   names=['uid', 'user_city', 'item_id', 'author_id', 'item_city', 'channel', 'finish', 'like', 'music_id',
                    'device_id', 'create_time', 'video_duration'],
                   header=None)


# %%
train_valid_set, test_set = train_test_split(data, test_size=0.1, random_state=42)
train_set, valid_set = train_test_split(train_valid_set, test_size=0.1, random_state=42)

# %%
train_set.to_csv('./train.csv',index=None)
valid_set.to_csv('./valid.csv',index=None)
test_set.to_csv('./test.csv',index=None)
