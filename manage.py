# -*- coding: utf-8 -*-

'''
manage script for retrieve data from the folder to create the data frame 
'''
#%%
import os
import config as conf
import pandas as pd 
#%%
def load_images(data_path):
    images_df=[]
    for class_name in os.listdir(data_path):
        sub_path=data_path+class_name+'/'
        for subimge in os.listdir(sub_path):
            if subimge != 'for_sift':
                subsub_path=sub_path+subimge+'/'
                for char_number in os.listdir(subsub_path):
                    number_path=subsub_path+char_number+'/'
                    for img in os.listdir(number_path):
                        image_path=number_path+img
                        full_class_name=class_name+'_'+conf.CHAR_LIST[char_number]
                        tmp = pd.DataFrame([image_path, full_class_name]).T
                        images_df.append(tmp)
    
    images_df = pd.concat(images_df, axis=0, ignore_index=True)
    images_df.columns = ['image', 'target']
    return images_df


#%%
if __name__ == '__main__':
    data_collection=load_images(conf.DATA_PATH)
    print(data_collection.head())
