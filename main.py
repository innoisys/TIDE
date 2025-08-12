import numpy as np
import load_dataset as ld
import os
import misc
from tensorflow import keras
import tide as v
from PIL import Image
import imageio
import hashlib

#### Arguments and CONSTANTS
dataset_path = misc.get_arg('dataset_path', './data/kid_dataset/')
cache_key = hashlib.md5('test'.encode()).hexdigest()
CACHE_DIR = misc.get_arg('cache', './cache')
CACHE_FILE = os.path.join(CACHE_DIR, f'{cache_key}-data.pckl')
OUTPUT_PATH = misc.get_arg('output', './output.ms-abnormal')
BATCH_SIZE = int(misc.get_arg('batch_size', 32))
OUTPUT_WEIGHT_PATH = os.path.join(OUTPUT_PATH, 'weights/{epoch:02d}.hdf5')
misc.ensure_path(OUTPUT_PATH)
misc.ensure_path(os.path.dirname(OUTPUT_WEIGHT_PATH))
misc.ensure_path(CACHE_DIR)


#### LOAD THE DATA
def load_data(dataset_path):
    x_train, y_train = ld.load_dataset_kid(size=(224, 224), dataset_path=dataset_path,
                                           excludes=[
                                               'ampulla',
                                               # 'inflammatory',
                                               'polypoid',
                                               'normalstom',
                                               'normaleso',
                                               'normalcolon',
                                               'vascular',
                                               # 'normalsb'
                                           ])
    x_normal = np.array([x_train[i] for i in range(0, len(x_train)) if y_train[i] == 0])
    x_abnormal = np.array([x_train[i] for i in range(0, len(x_train)) if y_train[i] == 1])
    return x_normal, x_abnormal


x_normal, x_abnormal = misc.cache(CACHE_FILE, lambda: load_data(dataset_path))
input_shape = (96, 96, 3)
latent_space = 6
x_train = []
tmp = x_abnormal if misc.get_arg('type', 'abnomral') else x_normal
for image in tmp:
    im = Image.fromarray(image)
    im = im.resize((input_shape[0], input_shape[1]))
    x_train.append(np.array(im))
x_train = np.array(x_train)
x_train = x_train.astype('float32') / 255.0

# train
vae = v.VAE(v.create_encoder(latent_space, input_shape), v.create_decoder(latent_space))
vae.build(input_shape=(None,) + input_shape)
vae.compile(optimizer=keras.optimizers.Adam())
cbs = [
    misc.CheckpointCallback(200, OUTPUT_WEIGHT_PATH)
]
vae.fit(x_train, epochs=50000, batch_size=BATCH_SIZE, callbacks=cbs, shuffle=True)


"""
    When trained, load the model weights and execute the following 
"""
decoder = vae.decoder
random_latent_vectors = np.random.normal(size=(BATCH_SIZE, latent_space))
generated_images = decoder.predict(random_latent_vectors)
for i, image in enumerate(generated_images):
    image_scaled = (image * 255).astype(np.uint8)
    filename = f"generated_image_{i+1}.png"
    filepath = os.path.join(OUTPUT_PATH, filename)
    imageio.imsave(filepath, image_scaled)

