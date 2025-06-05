def create_dueling_q_model(input_shape, num_actions, noise_type='gaussian', use_noisy_layer=True):
    """
    Skapar en Dueling DQN-modell med NoisyDense-lager för utforskning.

    Argument:
        input_shape (tuple): Form på indata (t.ex. (84, 84, 4) för Atari).
        num_actions (int): Antal möjliga handlingar i miljön.
        noise_type (str): Typ av brus att använda ('gaussian' eller 'uniform'). Standard är 'gaussian'.
        use_noisy_layer (bool): Om brusiga lager eller vanliga dense-lager ska användas. Standard är True.

    Returnerar:
        keras.Model: Kompilerad Dueling DQN-modell.
    """
    # Input lager för bilddata
    inputs = layers.Input(shape=input_shape)

    # Konvilutioonella lager för att extrahera funktioner från bilddata
    x = layers.Conv2D(32, 8, strides=4, activation="relu")(inputs)
    x = layers.Conv2D(64, 4, strides=2, activation="relu")(x)
    x = layers.Conv2D(64, 3, strides=1, activation="relu")(x)
    x = layers.Flatten()(x)

    # NoisyDense layer for improved exploration
    x = NoisyDense(512, sigma_init=0.5, noise_type=noise_type, use_noisy_layer=use_noisy_layer)(x)

    # dueling arkitektur
    value = NoisyDense(1, sigma_init=0.5, noise_type=noise_type, use_noisy_layer=use_noisy_layer)(x)              # State-value stream
    advantage = NoisyDense(num_actions, sigma_init=0.5, noise_type=noise_type, use_noisy_layer=use_noisy_layer)(x) # Advantage stream

    # kombinera värdena för att få Q-värden
    # Q(s, a) = V(s) + (A(s, a) - mean(A(s, ·)))
    q_values = layers.Lambda(
        lambda a: a[0] + (a[1] - tf.reduce_mean(a[1], axis=1, keepdims=True))
    )([value, advantage])

    # bygg modellen
    return models.Model(inputs=inputs, outputs=q_values)
