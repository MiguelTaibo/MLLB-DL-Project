from keras.layers import Conv2D, Dropout, Flatten, MaxPooling2D, Input, Dense, InputLayer
from keras.constraints import maxnorm
from keras.models import Model
from keras.regularizers import l2
def VGG16(dropout, num_classes=10, img_width=32, img_height=32, img_channels=3,l2_reg=0):
    input_image = Input(shape=(img_width,img_height,img_channels))
    x1 = Conv2D(64, (3, 3),padding='same', activation='relu',kernel_initializer="glorot_uniform",kernel_regularizer=l2(l2_reg),bias_regularizer=l2(l2_reg))(input_image)
    x1 = Conv2D(64, (3, 3),padding='same', activation='relu',kernel_initializer="glorot_uniform",kernel_regularizer=l2(l2_reg),bias_regularizer=l2(l2_reg))(x1)
    x1 = MaxPooling2D((2, 2))(x1)

    x2 = Conv2D(128, (3, 3),padding='same', activation='relu',kernel_initializer="glorot_uniform",kernel_regularizer=l2(l2_reg),bias_regularizer=l2(l2_reg))(x1)
    x2 = Conv2D(128, (3, 3),padding='same', activation='relu',kernel_initializer="glorot_uniform",kernel_regularizer=l2(l2_reg),bias_regularizer=l2(l2_reg))(x2)
    x2 = MaxPooling2D((2, 2))(x2)

    x3 = Conv2D(256, (3, 3),padding='same', activation='relu',kernel_initializer="glorot_uniform",kernel_regularizer=l2(l2_reg),bias_regularizer=l2(l2_reg))(x2)
    x3 = Conv2D(256, (3, 3),padding='same', activation='relu',kernel_initializer="glorot_uniform",kernel_regularizer=l2(l2_reg),bias_regularizer=l2(l2_reg))(x3)
    x3 = Conv2D(256, (3, 3),padding='same', activation='relu',kernel_initializer="glorot_uniform",kernel_regularizer=l2(l2_reg),bias_regularizer=l2(l2_reg))(x3)
    x3 = MaxPooling2D((2, 2))(x3)

    x4 = Conv2D(512, (3, 3),padding='same', activation='relu',kernel_initializer="glorot_uniform",kernel_regularizer=l2(l2_reg),bias_regularizer=l2(l2_reg))(x3)
    x4 = Conv2D(512, (3, 3),padding='same', activation='relu',kernel_initializer="glorot_uniform",kernel_regularizer=l2(l2_reg),bias_regularizer=l2(l2_reg))(x4)
    x4 = Conv2D(512, (3, 3),padding='same', activation='relu',kernel_initializer="glorot_uniform",kernel_regularizer=l2(l2_reg),bias_regularizer=l2(l2_reg))(x4)
    x4 = MaxPooling2D((2, 2))(x4)

    x5 = Conv2D(512, (3, 3),padding='same', activation='relu',kernel_initializer="glorot_uniform",kernel_regularizer=l2(l2_reg),bias_regularizer=l2(l2_reg))(x4)
    x5 = Conv2D(512, (3, 3),padding='same', activation='relu',kernel_initializer="glorot_uniform",kernel_regularizer=l2(l2_reg),bias_regularizer=l2(l2_reg))(x5)
    x5 = Conv2D(512, (3, 3),padding='same', activation='relu',kernel_initializer="glorot_uniform",kernel_regularizer=l2(l2_reg),bias_regularizer=l2(l2_reg))(x5)
    x5 = MaxPooling2D((2, 2))(x5)

    x6 = Flatten()(x5)
    x6=Dropout(dropout)(x6)
    x6=Dense(512, activation='relu', kernel_constraint=maxnorm(3),kernel_initializer="glorot_uniform",kernel_regularizer=l2(l2_reg),bias_regularizer=l2(l2_reg))(x6)
    x6=Dropout(dropout)(x6)
    x6=Dense(512, activation='relu', kernel_constraint=maxnorm(3),kernel_initializer="glorot_uniform",kernel_regularizer=l2(l2_reg),bias_regularizer=l2(l2_reg))(x6)
    x=Dropout(dropout)(x6)
    out= Dense(num_classes, activation='softmax',kernel_initializer="glorot_uniform",kernel_regularizer=l2(l2_reg),bias_regularizer=l2(l2_reg))(x)

    model = Model(inputs = input_image, outputs = out)
    
    return model

def VGG19(dropout, num_classes=10, img_width=32, img_height=32, img_channels=3,l2_reg=0):
    input_image = Input(shape=(img_width,img_height,img_channels))
    x1 = Conv2D(64, (3, 3),padding='same', activation='relu',kernel_initializer="glorot_uniform",kernel_regularizer=l2(l2_reg),bias_regularizer=l2(l2_reg))(input_image)
    x1 = Conv2D(64, (3, 3),padding='same', activation='relu',kernel_initializer="glorot_uniform",kernel_regularizer=l2(l2_reg),bias_regularizer=l2(l2_reg))(x1)
    x1 = MaxPooling2D((2, 2))(x1)

    x2 = Conv2D(128, (3, 3),padding='same', activation='relu',kernel_initializer="glorot_uniform",kernel_regularizer=l2(l2_reg),bias_regularizer=l2(l2_reg))(x1)
    x2 = Conv2D(128, (3, 3),padding='same', activation='relu',kernel_initializer="glorot_uniform",kernel_regularizer=l2(l2_reg),bias_regularizer=l2(l2_reg))(x2)
    x2 = MaxPooling2D((2, 2))(x2)

    x3 = Conv2D(256, (3, 3),padding='same', activation='relu',kernel_initializer="glorot_uniform",kernel_regularizer=l2(l2_reg),bias_regularizer=l2(l2_reg))(x2)
    x3 = Conv2D(256, (3, 3),padding='same', activation='relu',kernel_initializer="glorot_uniform",kernel_regularizer=l2(l2_reg),bias_regularizer=l2(l2_reg))(x3)
    x3 = Conv2D(256, (3, 3),padding='same', activation='relu',kernel_initializer="glorot_uniform",kernel_regularizer=l2(l2_reg),bias_regularizer=l2(l2_reg))(x3)
    x3 = Conv2D(256, (3, 3),padding='same', activation='relu',kernel_initializer="glorot_uniform",kernel_regularizer=l2(l2_reg),bias_regularizer=l2(l2_reg))(x3)
    x3 = MaxPooling2D((2, 2))(x3)

    x4 = Conv2D(512, (3, 3),padding='same', activation='relu',kernel_initializer="glorot_uniform",kernel_regularizer=l2(l2_reg),bias_regularizer=l2(l2_reg))(x3)
    x4 = Conv2D(512, (3, 3),padding='same', activation='relu',kernel_initializer="glorot_uniform",kernel_regularizer=l2(l2_reg),bias_regularizer=l2(l2_reg))(x4)
    x4 = Conv2D(512, (3, 3),padding='same', activation='relu',kernel_initializer="glorot_uniform",kernel_regularizer=l2(l2_reg),bias_regularizer=l2(l2_reg))(x4)
    x4 = Conv2D(512, (3, 3),padding='same', activation='relu',kernel_initializer="glorot_uniform",kernel_regularizer=l2(l2_reg),bias_regularizer=l2(l2_reg))(x4)
    x4 = MaxPooling2D((2, 2))(x4)

    x5 = Conv2D(512, (3, 3),padding='same', activation='relu',kernel_initializer="glorot_uniform",kernel_regularizer=l2(l2_reg),bias_regularizer=l2(l2_reg))(x4)
    x5 = Conv2D(512, (3, 3),padding='same', activation='relu',kernel_initializer="glorot_uniform",kernel_regularizer=l2(l2_reg),bias_regularizer=l2(l2_reg))(x5)
    x5 = Conv2D(512, (3, 3),padding='same', activation='relu',kernel_initializer="glorot_uniform",kernel_regularizer=l2(l2_reg),bias_regularizer=l2(l2_reg))(x5)
    x5 = Conv2D(512, (3, 3),padding='same', activation='relu',kernel_initializer="glorot_uniform",kernel_regularizer=l2(l2_reg),bias_regularizer=l2(l2_reg))(x5)
    x5 = MaxPooling2D((2, 2))(x5)

    x6 = Flatten()(x5)
    x6=Dropout(dropout)(x6)
    x6=Dense(512, activation='relu', kernel_constraint=maxnorm(3),kernel_initializer="glorot_uniform",kernel_regularizer=l2(l2_reg),bias_regularizer=l2(l2_reg))(x6)
    x6=Dropout(dropout)(x6)
    x6=Dense(512, activation='relu', kernel_constraint=maxnorm(3),kernel_initializer="glorot_uniform",kernel_regularizer=l2(l2_reg),bias_regularizer=l2(l2_reg))(x6)
    x=Dropout(dropout)(x6)
    out= Dense(num_classes, activation='softmax')(x)

    model = Model(inputs = input_image, outputs = out)
    
    return model

def MLP_model(dropout, img_width=32, img_height=32, img_channels=3, num_classes=10,l2_reg=0,n_hidden=2):
    input_image = Input(shape=(img_width,img_height,img_channels))
    x = Flatten()(input_image)
    
    for layer in reversed(range(n_hidden)):
        x = Dense(int((img_height*img_width+num_classes)*(layer+1)/(n_hidden+1)), activation='relu', kernel_constraint=maxnorm(3),kernel_initializer="glorot_uniform",kernel_regularizer=l2(l2_reg),bias_regularizer=l2(l2_reg))(x)
    x=Dropout(dropout)(x)
    out= Dense(num_classes, activation='softmax',kernel_initializer="glorot_uniform",kernel_regularizer=l2(l2_reg),bias_regularizer=l2(l2_reg))(x)
    model = Model(inputs = input_image, outputs = out)
    
    return model