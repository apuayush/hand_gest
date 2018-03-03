from modules import *

img_rows, img_cols = 200, 200
img_channels = 1

batch_size = 32


nb_classes = 6

nb_epoch = 15  #25

# Total number of convolutional filters to use
nb_filters = 32
# Max pooling
nb_pool = 2
# Size of convolution kernel
nb_conv = 3

def custom_model_hand():
    '''
    USER CODE STARTS HERE
    '''
    model = Sequential()


    model.add(Conv2D(nb_filters, (nb_conv, nb_conv),
                        padding='valid',
                        input_shape=(img_rows, img_cols, img_channels)))
    convout1 = Activation('relu')
    model.add(convout1)
    model.add(Conv2D(nb_filters, (nb_conv, nb_conv)))
    convout2 = Activation('relu')
    model.add(convout2)
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    """
    image_model.add(Flatten())

    image_model.add(Dense(512))
    image_model.add(Activation('tanh'))
    image_model.add(Dropout(0.2))
    
    image_model.add(Dense(512))
    image_model.add(Activation('tanh'))
    image_model.add(Dropout(0.15))
    
    image_model.add(Dense(512))
    image_model.add(Activation('tanh'))
    image_model.add(Dropout(0.1))
    
    image_model.add(Dense(512))
    image_model.add(Activation('tanh'))
    
    image_model.add(Dense(512))
    image_model.add(Activation('tanh'))
    
    image_model.add(Dense(512))
    image_model.add(Activation('tanh'))
    
    image_model.add(Dense(512))
    image_model.add(Activation('tanh'))
    
    image_model.add(Dense(512))
    image_model.add(Activation('tanh'))
    
    image_model.add(Dense(6))
    image_model.add(Activation('sigmoid'))
    """
    
    return model

def make_model(file):
    print("==================================================") 
    
    print("Creating Model At: ",file) 
    start_time = time.time()
    model = custom_model_hand() 
    
    plot_model(model, to_file='model.png', show_shapes= True, show_layer_names = True)
    
    json_model = model.to_json()
    
    with open(file, "w") as json_file:
        json_file.write(json_model)
    
    end_time = time.time()
    total_time = end_time-start_time
    print("Model Created: ",total_time, " seconds")
    
    print("==================================================")
    

if __name__ == "__main__":   
    make_model("hand_detection_model_3.json")
    
