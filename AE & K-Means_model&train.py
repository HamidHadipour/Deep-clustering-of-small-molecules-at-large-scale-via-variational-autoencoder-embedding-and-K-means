import tensorflow as tf
import tensorflow.keras as tfk
import tensorflow.keras.layers as tfkl
import numpy as np
from keras import regularizers
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.initializers import glorot_uniform

tf.random.set_seed(54)
x_train, x_test = train_test_split(
    cleaned, test_size=0.3, random_state=1)

#x_train = x_train/10.66  # 
#x_test = x_test/10.66

x_train = np.array(x_train)
x_test = np.array(x_test) 

# Set random seed
tf.random.set_seed(54) # 

# Encoder
inputs = tfkl.Input(shape = (243))
x = inputs
#x = tfkl.Dense(512,'relu')(x)
#x = tfkl.Dense(64,'relu')(x)
#x = tfkl.Dense(32,'relu')(x)
x = tfkl.Dense(16,'relu')(x)
encoder = tfk.Model(inputs = inputs, outputs = x)

# Decoder
inputs = tfkl.Input(shape = (16))
x = inputs
#x = tfkl.Dense(32,'relu')(x)
#x = tfkl.Dense(128,'relu')(x)
#x = tfkl.Dense(512,'relu')(x)
x = tfkl.Dense(243*1,'relu')(x)
decoder = tfk.Model(inputs = inputs, outputs = x)

# Autoencoder
inputs = tfkl.Input(shape = (243))
x = encoder(inputs)
x = decoder(x)
model = tfk.Model(inputs = inputs, outputs = x)

# Train the Autoencoder
model.compile(loss = 'mse',optimizer = 'adam')
model.fit(x_train, x_train,epochs = 10,batch_size = 32)



latent = encoder.predict(cleaned)

encoder.save('pca50_encoder32_10epoch.h5')



