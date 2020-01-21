# -*- coding: utf-8 -*-

#%%
#importing libraries
import manage as mg
import config as conf
import process as pro
import model as model
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
import matplotlib.pyplot as plt
#%%
if __name__ == '__main__':
    images_dataFrame=mg.load_images(conf.DATA_PATH)
    X_train, X_test, y_train, y_test =pro.get_train_test_target(images_dataFrame.sample(n=100))
    dataset_train=pro.CreateDataset(data_collection=X_train,image_size=50)
    dataset_test=pro.CreateDataset(data_collection=X_test,image_size=50)
    encoder=LabelEncoder()
    encoding=encoder.fit(y_train)
    print(encoding.classes_)
    transformed=encoder.transform(y_train)
    print(transformed)
    Y_Train = np_utils.to_categorical(transformed, 196)
    
    
    encoding_y=encoder.fit(y_test)
    print(encoding_y.classes_)
    transformed_test=encoder.transform(y_test)
    print(transformed_test)
    Y_Test= np_utils.to_categorical(transformed_test, 196)
    
    model=model.build_model()
    model.summary()
    hist = model.fit(x=dataset_train,y=Y_Train, epochs=10, batch_size=128, validation_data=(dataset_test, Y_Test), verbose=1)
    test_score = model.evaluate(dataset_test,Y_Test)
    print('Test loss {:.4f}, accuracy {:.2f}%'.format(test_score[0], test_score[1] * 100))

    f, ax = plt.subplots()
    ax.plot([None] + hist.history['acc'], 'o-')
    ax.plot([None] + hist.history['val_acc'], 'x-')
    # Plot legend and use the best location automatically: loc = 0.
    ax.legend(['Train acc', 'Validation acc'], loc = 0)
    ax.set_title('Training/Validation acc per Epoch')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('acc')
    



