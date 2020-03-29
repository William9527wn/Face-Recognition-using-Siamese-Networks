from data_preparation import get_data
from model import build_base_network
from sklearn.model_selection import train_test_split
from keras.optimizers import RMSprop
from keras.layers import Input, Lambda
from keras import backend as K
from keras.models import Model


def euclidean_distance(vects):
    x,y =vects
    return K.sqrt(K.sum(K.square(x-y), axis=1, keepdims=True))

def eucl_dist_out_shapes(shapes):
    shape1, shape2 = shapes
    return shape1[0], 1

def contrastive_loss(y_true, y_pred):
    margin = 1
    return K.mean(y_true * K.square(y_pred)+ (1- y_true) * K.square(K.maximum(margin - y_pred, 0)))

def compute_accuracy(predictions, labels):
    return labels[predictions.ravel()<0.5].mean()

size = 2
total_sample_size = 10000

X,Y = get_data(size, total_sample_size)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size= 0.25)


input_dim = x_train.shape[2:]
img_a = Input(shape=input_dim)
img_b = Input(shape=input_dim)

base_network = build_base_network(input_dim)
feat_vect_a = base_network(img_a)
feat_vect_b = base_network(img_b)

distance = Lambda(euclidean_distance, output_shape=eucl_dist_out_shapes)([feat_vect_a, feat_vect_b])

epochs = 15
rms = RMSprop()
model = Model(input=[img_a, img_b], output = distance)

model.compile(loss=contrastive_loss, optimizer=rms)

img_1 = x_train[:, 0]
img_2 = x_train[:, 1]

model.fit([img_1, img_2], y_train, validation_split= 0.25, 
          batch_size=128, verbose=2, epochs=epochs )

pred = model.predict([x_test[:,0], x_test[:,1]])

print('Model achived accuracy of {} in {} epochs'.format(compute_accuracy(pred, y_test), epochs))