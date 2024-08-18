import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate

#定义模型
def unet_models():
    #输入层
    inputs=Input(shape(256,256,1))

    #编码器
    conv1=Conv2D(64,3,activation="relu",padding="same")(iputs)
    conv1=Conv2D(64,3,activation="relu",padding="same")(conv1)
    pool1=MaxPooling2D(pool_size=(2,2))(conv1)

    conv2=Conv2D(128,3,activation="relu",padding="same")(pool1)
    conv2=Conv2D(128,3,activation="relu",padding="same")(conv2)
    pool2=MaxPooling2D(pool_size=(2,2))(conv2)

    conv3=Conv2D(256,3,activation="relu",padding="same")(pool2)
    conv3=Conv2D(256,3,activation="relu",padding="same")(conv3)
    pool3=MaxPooling2D(pool_size=(2,2))(conv3)

    conv4=Conv2D(512,3,activation="relu",padding="same")(pool3)
    conv4=Conv2D(512,3,activation="relu",padding="same")(conv4)
    pool4=MaxPooling2D(pool_size=(2,2))(conv4)

    conv5=Conv2D(1024,3,activation="relu",padding="same")(pool4)
    conv5=Conv2D(1024,3,activation="relu",padding="same")(conv5)
   
    #解码器

    up6 = UpSampling2D(size=(2, 2))(conv5)
    up6=Conv2D(512,2,activation="relu",padding="same")(up6)
    merge6=Concatenate()([conv4,up6])
    conv6 = Conv2D(512, 3, activation='relu', padding='same')(merge6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same')(conv6)

    up7 = UpSampling2D(size=(2, 2))(conv6)
    up6=Conv2D(256,2,activation="relu",padding="same")(up7)
    merge7=Concatenate()([conv3,up7])
    conv7= Conv2D(256, 3, activation='relu', padding='same')(merge7)
    conv7= Conv2D(256, 3, activation='relu', padding='same')(conv7)


    up8 = UpSampling2D(size=(2, 2))(conv7)
    up8=Conv2D(128,2,activation="relu",padding="same")(up8)
    merge8=Concatenate()([conv2,up8])
    conv8= Conv2D(128, 3, activation='relu', padding='same')(merge8)
    conv8= Conv2D(128, 3, activation='relu', padding='same')(conv8)


    up9 = UpSampling2D(size=(2, 2))(conv8)
    up9=Conv2D(128,2,activation="relu",padding="same")(up9)
    merge9=Concatenate()([conv1,up9])
    conv9= Conv2D(128, 3, activation='relu', padding='same')(merge9)
    conv9= Conv2D(128, 3, activation='relu', padding='same')(conv9)

    outputs = Conv2D(1, 1, activation='sigmoid')(conv9)
  
    model = Model(inputs=inputs, outputs=outputs)
    return model

    