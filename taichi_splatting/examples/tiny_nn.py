import tinycudann as tcnn


def tiny_nn(hidden:int, layers:int, num_features:int, output_features:int):
  return tcnn.NetworkWithInputEncoding(
    num_features, output_features,
    encoding_config=dict(
      otype = "Identity",
    ), 
    network_config = dict(
      otype = "FullyFusedMLP",
      activation = "ReLU",
      output_activation = "None",
      n_neurons = hidden,
      n_hidden_layers = layers,
    )
  )