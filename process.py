# -*- coding: utf-8 -*-

#%%
from sklearn.model_selection import train_test_split
from PIL import Image
import numpy as np 
#%%
def get_train_test_target(df):

	X_train, X_test, y_train, y_test = train_test_split(df['image'],
                                                     df['target'],
                                                     test_size=0.20,
                                                     random_state=101)

	X_train.reset_index(drop=True, inplace=True)
	X_test.reset_index(drop=True, inplace=True)

	y_train.reset_index(drop=True, inplace=True)
	y_test.reset_index(drop=True, inplace=True)

	return X_train, X_test, y_train, y_test

#%%
def _im_resize(df, n, image_size):
    im = Image.open(df[n])
    im_resized = im.resize((image_size, image_size))
    im_asarray=np.array(im_resized)
    return im_asarray


#%%
def CreateDataset(data_collection,image_size):
    if data_collection.shape == (1,2):#it has one video
        data_collection=data_collection['image']
        data_collection.reset_index(drop=True, inplace=True)
    X=data_collection.to_frame()
    tmp = np.zeros((len(X),image_size,image_size,3),dtype='float32')

    for n in range(0, len(X)):
        vid = _im_resize(X['image'], n,image_size)
        tmp[n] = vid
    reshaped_tmp=tmp.reshape(tmp.shape[0],tmp.shape[1],tmp.shape[2],3)
    return reshaped_tmp

