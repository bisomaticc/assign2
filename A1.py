from keras.datasets import mnist
import matplotlib.pyplot as plt
from keras.utils.np_utils import to_categorical
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import RMSprop

dataset=mnist.load_data('mymnist.db')

train,test=dataset

#training dataset is divided in input data and output data
trainx,trainy=train

#testing dataset is divided in input data and output data
testx,testy=test

#Actually the dataset contains images of size 28X28. So we are converting this image data from
#2D to 1D by using this reshape
X_train_1d = trainx.reshape(-1 , 28*28)
X_test_1d = testx.reshape(-1 , 28*28)

trainx = X_train_1d.astype('float32')
testx = X_test_1d.astype('float32')

#categorising our data
trainycat=to_categorical(trainy)

#creating our model
model=Sequential()
model.add(Dense(units=256,input_dim=784,activation='relu'))
model.add(Dense(units=64,activation='relu'))
model.add(Dense(units=10,activation='softmax'))
model.compile(optimizer=RMSprop(), loss='categorical_crossentropy', 
             metrics=['accuracy'])
h = model.fit(trainx, trainycat, epochs=1,verbose=0)

#printing accuracy
print(int(h.history['accuracy'][-1]*100))
