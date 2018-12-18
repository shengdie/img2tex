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
        enc_lstm_dim = 256
        batch_size = self._config.batch_size
        #batch_size = img.get_shape().as_list()[0]
        #feat_size = 64
        #dec_lstm_dim = 512
        #vocab_size = 503
        #embedding_size = 80
        img = tf.cast(img, tf.float32) / 255.
        with tf.variable_scope("convolutional_encoder"):
            out = tf.layers.conv2d(img, 256, 3, 1, "SAME",
                    activation=tf.nn.relu)
            out = tf.layers.max_pooling2d(out, 2, 2, "SAME")

            # conv + max pool -> /2
            out = tf.layers.conv2d(out, 128, 3, 1, "SAME",
                    activation=tf.nn.relu)
            out = tf.layers.max_pooling2d(out, 2, 2, "SAME")
            out = tf.layers.conv2d(out, 128, 3, 1, "SAME",
                    activation=tf.nn.relu)
            # regular conv -> id
            out = tf.layers.conv2d(out, 64, 3, 1, "SAME",
                    activation=tf.nn.relu)
            out = tf.layers.max_pooling2d(out, 2, 2, "SAME")

        #if self._config.positional_embeddings:
               # from tensor2tensor lib - positional embeddings
        #    out = add_timing_signal_nd(out)
        def fn(inp):
            enc_init_shape = [batch_size, enc_lstm_dim]
            with tf.variable_scope('encoder_rnn'):
                with tf.variable_scope('forward'):
                    lstm_cell_fw = tf.nn.rnn_cell.LSTMCell(enc_lstm_dim)
                    init_fw = tf.nn.rnn_cell.LSTMStateTuple( \
                        tf.get_variable("enc_fw_c", enc_init_shape), \
                        tf.get_variable("enc_fw_h", enc_init_shape)
                    )
                with tf.variable_scope('backward'):
                    lstm_cell_bw = tf.nn.rnn_cell.LSTMCell(enc_lstm_dim)
                    init_bw = tf.nn.rnn_cell.LSTMStateTuple( \
                        tf.get_variable("enc_bw_c", enc_init_shape), \
                        tf.get_variable("enc_bw_h", enc_init_shape)
                    )
                output, _ = tf.nn.bidirectional_dynamic_rnn(lstm_cell_fw, \
                                                            lstm_cell_bw, \
                                                            inp, \
                                                            sequence_length=tf.fill([batch_size], \
                                                                                    tf.shape(inp)[1]), \
                                                            initial_state_fw=init_fw, \
                                                            initial_state_bw=init_bw \
                                                            )
            return tf.concat(output, 2)

        fun = tf.make_template('fun', fn)
        rows_first = tf.transpose(out, [1, 0, 2, 3])
        res = tf.map_fn(fun, rows_first, dtype=tf.float32)
        out = tf.transpose(res, [1, 0, 2, 3])

        
        if self._config.positional_embeddings:
                # from tensor2tensor lib - positional embeddings
            out = add_timing_signal_nd(out)

        return out

            #out = tf.layers.conv2d(out, 64, 3, 1, "SAME",
            #        activation=tf.nn.relu)

            #if self._config.encoder_cnn == "vanilla":
            #out = tf.layers.max_pooling2d(out, (2, 4), (2, 2), "SAME")
            #out = tf.layers.conv2d(out, 256, 3, 1, "VALID",
            #        activation=tf.nn.relu)