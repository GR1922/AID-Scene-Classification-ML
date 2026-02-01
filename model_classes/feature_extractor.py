import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input


class EfficientNetB0FeatureExtractor:
    """
    EfficientNet-B0 preaddestrata su ImageNet usata come feature extractor.
    Output: vettore 1280D per immagine (pooling avg).
    """

    def __init__(self, img_size: int = 224):
        self.img_size = img_size

        self.model = EfficientNetB0(
            include_top=False,
            weights="imagenet",
            input_shape=(img_size, img_size, 3),
            pooling="avg"
        )
        self.model.trainable = False

    def preprocess(self, img: tf.Tensor) -> tf.Tensor:
        img = tf.image.resize(img, (self.img_size, self.img_size))
        img = tf.cast(img, tf.float32)
        img = preprocess_input(img)
        return img

    def extract(self, batch_imgs: tf.Tensor) -> tf.Tensor:
        return self.model(batch_imgs, training=False)
