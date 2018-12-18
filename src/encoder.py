import tensorflow as tf
from .general.positional import add_timing_signal_nd


class Encoder(object):
    """Encoder
    """
    def __init__(self, config):
        self._config = config

    def __call__(self, training, img, dropout):
        """Applies convolutions to the image
        Args:
            training: (tf.placeholder) tf.bool
            img: batch of img, shape = (?, height, width, channels), of type
                tf.uint8
        Returns:
            the encoded images, shape = (?, h', w', c')
        """
        img = tf.cast(img, tf.float32) / 255.

        with tf.variable_scope("convolutional_encoder"):
            # conv + max pool -> /2
            """
            # encoder 2
            out = tf.layers.conv2d(img, 128, 3, 1, "SAME",
                    activation=tf.nn.relu)
            out = tf.layers.max_pooling2d(out, 2, 2, "SAME")

            # conv + max pool -> /2
            out = tf.layers.conv2d(out, 128, 3, 1, "SAME",
                    activation=tf.nn.relu)
            #out = tf.layers.max_pooling2d(out, 2, 2, "SAME")
            #out = tf.layers.conv2d(out, 128, 3, 1, "SAME",
            #        activation=tf.nn.relu)
            #out = tf.layers.max_pooling2d(out, 2, 2, "SAME")
            # regular conv -> id
            out = tf.layers.conv2d(out, 256, 3, 1, "SAME",
                    activation=tf.nn.relu)
            out = tf.layers.max_pooling2d(out, (2,4), 2, "SAME")

            out = tf.layers.conv2d(out, 256, 3, 1, "SAME",
                    activation=tf.nn.relu)

            #if self._config.encoder_cnn == "vanilla":
            out = tf.layers.max_pooling2d(out, (2, 4), (2, 2), "SAME")
            out = tf.layers.conv2d(out, 256, 3, 1, "VALID",
                    activation=tf.nn.relu)
            # end encocer2
            """
            #out = tf.layers.conv2d(out, 512, 3, 1, "SAME",
            #        activation=tf.nn.relu)
            """
            # encoder 3
            out = tf.layers.conv2d(img, 32, 3, 1, "SAME",
                    activation=tf.nn.relu)
            out = tf.layers.conv2d(out, 32, 3, 1, "SAME",
                    activation=tf.nn.relu)
            out = tf.layers.max_pooling2d(out, 2, 2, "SAME")
            
            out = tf.layers.conv2d(out, 64, 3, 1, "SAME",
                    activation=tf.nn.relu)
            out = tf.layers.conv2d(out, 64, 3, 1, "SAME",
                    activation=tf.nn.relu)
            out = tf.layers.max_pooling2d(out, 2, 2, "SAME")

            #out = tf.layers.conv2d(img, 128, 3, 1, "SAME",
            #        activation=tf.nn.relu)
            #out = tf.layers.conv2d(img, 128, 3, 1, "SAME",
            #        activation=tf.nn.relu)
            out = tf.layers.conv2d(out, 128, 3, 1, "SAME",
                    activation=tf.nn.relu)
            out = tf.layers.conv2d(out, 128, 3, 1, "SAME",
                    activation=tf.nn.relu)
            out = tf.layers.max_pooling2d(out, 2, 2, "SAME")

            out = tf.layers.conv2d(out, 256, 3, 1, "SAME",
                    activation=tf.nn.relu)
            #out = tf.layers.conv2d(img, 256, 3, 1, "SAME",
            #        activation=tf.nn.relu)
            #out = tf.layers.conv2d(img, 256, 3, 1, "SAME",
            #        activation=tf.nn.relu)
            #out = tf.layers.conv2d(out, 256, 3, 1, "SAME",
            #        activation=tf.nn.relu)
            #out = tf.layers.max_pooling2d(out, 2, 2, "SAME")
            #out = tf.layers.conv2d(img, 256, 3, 1, "SAME",
            #        activation=tf.nn.relu)
            #out = tf.layers.conv2d(img, 256, 3, 1, "SAME",
            #        activation=tf.nn.relu)
            
            """
            #new_encoder1
            out = tf.layers.conv2d(img, 64, 3, 1, "SAME",
                    activation=tf.nn.relu)
            #out = tf.layers.max_pooling2d(out, 2, 2, "SAME")

            # conv + max pool -> /2
            out = tf.layers.conv2d(out, 128, 3, 1, "SAME",
                    activation=tf.nn.relu)
            out = tf.layers.max_pooling2d(out, 2, 2, "SAME")
            out = tf.layers.conv2d(out, 128, 3, 1, "SAME",
                    activation=tf.nn.relu)
            # regular conv -> id
            out = tf.layers.conv2d(out, 256, 3, 1, "SAME",
                    activation=tf.nn.relu)
            out = tf.layers.max_pooling2d(out, 2, 2, "SAME")

            out = tf.layers.conv2d(out, 256, 3, 1, "SAME",
                    activation=tf.nn.relu)

            #if self._config.encoder_cnn == "vanilla":
            out = tf.layers.max_pooling2d(out, (2, 4), (2, 2), "SAME")
            out = tf.layers.conv2d(out, 256, 3, 1, "VALID",
                    activation=tf.nn.relu)
            # end newencoder1
            #"""
            #if self._config.encoder_cnn == "vanilla":
            #    out = tf.layers.max_pooling2d(out, (1, 2), (1, 2), "SAME")

            #if self._config.encoder_cnn == "cnn":
                # conv with stride /2 (replaces the 2 max pool)
            #    out = tf.layers.conv2d(out, 512, (2, 4), 2, "SAME")

            # conv
            #out = tf.layers.conv2d(out, 512, 3, 1, "VALID",
            #        activation=tf.nn.relu)

            if self._config.positional_embeddings:
                # from tensor2tensor lib - positional embeddings
                out = add_timing_signal_nd(out)

        return out
