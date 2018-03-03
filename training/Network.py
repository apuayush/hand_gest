from modules import *
from data_handler import *
from sklearn.utils import shuffle

nb_classes = 6
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



from keras.models import model_from_json
model = None
with open('hand_detection_model_3.json') as json_file:
    model = model_from_json(json_file.read())
    json_file.close()
    
print(model)
from keras.utils.vis_utils import plot_model
plot_model(model, to_file='model.png', show_shapes= True, show_layer_names = True)

X_orig, Y_orig = load_test_image()
X_orig,Y_orig = shuffle(X_orig, Y_orig)
print(Y_orig)
X_train, X_test, Y_train, Y_test = train_test_split(X_orig, Y_orig, test_size = 0.2,random_state = 4)

weights_path = 'hand_detection_weights_3.h5'
learning_rate = 0.01
decay_rate = 0.000001
loss_function = "mean_squared_error"
momentum= 0.9
nesterov=True
optimizer = "SGD"

model.save_weights(weights_path)

opt = SGD(lr=learning_rate, decay=decay_rate, momentum=momentum, nesterov=nesterov)

model.compile(loss='categorical_crossentropy',optimizer="adam",metrics=['accuracy'])

def predict(inputs):
    pre = model.predict(inputs)
    return pre

model.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch,
                 verbose=1, validation_split=0.2)

# 96.41% accuracy and loss 0.2
model.evaluate(X_test, Y_test)