import tensorflow as tf
from tensorflow.keras import layers
import math

class NoisyDense(layers.Layer):
    def __init__(self, units, sigma_init=0.5, noise_type='gaussian', use_noisy_layer=True, **kwargs):
        """
        Noisy Dense-lager som beskrivs i 'Noisy Networks for Exploration'.
        
        Argument:
            units (int): Antal utgångsenheter.
            sigma_init (float): Initialvärde för standardavvikelsen på bruset.
            noise_type (str): Typ av brus ('gaussian' eller 'uniform').
            use_noisy_layer (bool): Om ett noisy dense-lager ska användas eller ett vanligt dense-lager för ablation.
        """
        super(NoisyDense, self).__init__(**kwargs)
        self.units = units
        self.sigma_init = sigma_init
        self.noise_type = noise_type
        self.use_noisy_layer = use_noisy_layer

    def build(self, input_shape):
        """
        Skapar lager-vikter: medelvärde och standardavvikelse för både vikter och bias.
        """
        input_dim = int(input_shape[-1])
        limit = 1 / math.sqrt(input_dim)

        # medelvärdesvikter initialiserade till ett litet intervall
        self.mu_w = self.add_weight(
            "mu_w",
            shape=(input_dim, self.units),
            initializer=tf.random_uniform_initializer(-limit, limit)
        )
        # standardavvikelsevikter initialiserade till sigma_init / sqrt(input_dim)
        self.sigma_w = self.add_weight(
            "sigma_w",
            shape=(input_dim, self.units),
            initializer=tf.constant_initializer(self.sigma_init / math.sqrt(input_dim))
        )

        # medelvärdesbias initialiserade till ett litet intervall
        self.mu_b = self.add_weight(
            "mu_b",
            shape=(self.units,),
            initializer=tf.random_uniform_initializer(-limit, limit)
        )
        # standardavvikelsebias initialiserade till sigma_init / sqrt(input_dim)
        self.sigma_b = self.add_weight(
            "sigma_b",
            shape=(self.units,),
            initializer=tf.constant_initializer(self.sigma_init / math.sqrt(input_dim))
        )
    
    def call(self, inputs, training=True):
        """
        Framåtpassering med valfri brusinsprutning under träning.
        Argument:
            inputs (Tensor): Indatatensor.
            training (bool): Om lagret ska injicera brus (True) eller inte (False).
        Returnerar:
            Tensor: Utdata från det brusiga dense-lagret.
        """
        dtype = inputs.dtype  # Stödjer mixed precision (float16 eller float32)
        if self.use_noisy_layer and training:
            if self.noise_type == 'gaussian':
                # Sampla brus för in- och utgångsdimensioner (Gaussiskt)
                epsilon_in = tf.random.normal((inputs.shape[-1], 1), dtype=dtype)
                epsilon_out = tf.random.normal((1, self.units), dtype=dtype)
                epsilon_w = tf.matmul(epsilon_in, epsilon_out)

                # Applicera brus på vikter och bias
                noisy_w = tf.cast(self.mu_w, dtype) + tf.cast(self.sigma_w, dtype) * epsilon_w
                noisy_b = tf.cast(self.mu_b, dtype) + tf.cast(self.sigma_b, dtype) * tf.squeeze(epsilon_out)
            elif self.noise_type == 'uniform':
                # Uniformt brusalternativ
                epsilon_in = tf.random.uniform((inputs.shape[-1], 1), dtype=dtype)
                epsilon_out = tf.random.uniform((1, self.units), dtype=dtype)
                epsilon_w = tf.matmul(epsilon_in, epsilon_out)

                # Applicera brus på vikter och bias
                noisy_w = tf.cast(self.mu_w, dtype) + tf.cast(self.sigma_w, dtype) * epsilon_w
                noisy_b = tf.cast(self.mu_b, dtype) + tf.cast(self.sigma_b, dtype) * tf.squeeze(epsilon_out)
            else:
                raise ValueError("Unsupported noise type. Choose either 'gaussian' or 'uniform'.")
        else:
            # Använd medelvärdesvikter och bias under utvärdering eller om `use_noisy_layer` är False
            noisy_w = tf.cast(self.mu_w, dtype)
            noisy_b = tf.cast(self.mu_b, dtype)

        # Standardberäkning för dense-lager
        return tf.matmul(inputs, noisy_w) + noisy_b

    def get_config(self):
        """
        Returnerar lagrets konfiguration för serialisering.
        """
        config = super(NoisyDense, self).get_config()
        config.update({
            "units": self.units,
            "sigma_init": self.sigma_init,
            "noise_type": self.noise_type,
            "use_noisy_layer": self.use_noisy_layer
        })
        return config

    @classmethod
    def from_config(cls, config):
        """
        Skapar ett lager från dess konfiguration.
        """
        return cls(**config)
