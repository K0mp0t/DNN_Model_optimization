from torch import nn
import torch


def get_layers_to_prune(model):
  children = list(model.children())
  return [model] if len(children) == 0 else [ci for c in children for ci in get_layers_to_prune(c)]



def prune_torch_model(layers_to_prune, fraction=0.1):
  sorted_indices = None
  prev_layer_output_size = 0
  prunable_counter = 0

  for i, layer in enumerate(layers_to_prune):
    # first process indices from previous layer
    if sorted_indices is not None:
      if isinstance(layer, nn.Conv2d):
        layer.in_channels = len(sorted_indices)
        layer.weight = torch.nn.Parameter(layer.weight[:, sorted_indices])
      elif isinstance(layer, nn.BatchNorm2d):
        layer.num_features = len(sorted_indices)

        layer.weight = torch.nn.Parameter(layer.weight[sorted_indices])
        layer.bias = torch.nn.Parameter(layer.bias[sorted_indices])
        layer.running_mean = layer.running_mean[sorted_indices]
        layer.running_var = layer.running_var[sorted_indices]
      elif isinstance(layer, nn.LSTM):
        # its here due to conv extracted features concatenation with second input
        sorted_indices = list(set(torch.arange(layer.input_size).tolist()) - (set(torch.arange(prev_layer_output_size).tolist()) - set(sorted_indices.numpy())))

        for attr in dir(layer):
          if attr.startswith('weight_ih'):
            layer.input_size = len(sorted_indices)
            setattr(layer, attr, torch.nn.Parameter(getattr(layer, attr)[:, sorted_indices]))

      elif isinstance(layer, nn.Linear):
        layer.in_features = len(sorted_indices)
        layer.weight = torch.nn.Parameter(layer.weight[:, sorted_indices])

    if len(layers_to_prune) - i <= 1:
      break

    # third prune
    if isinstance(layer, nn.Conv2d):
      l1_kernelwise = torch.sum(torch.abs(layer.weight), dim=(1, 2, 3)).detach()
      sorted_indices = torch.argsort(l1_kernelwise)[int(fraction*layer.out_channels):]
      sorted_indices = torch.sort(sorted_indices).values

      prev_layer_output_size = layer.out_channels
      layer.out_channels = len(sorted_indices)
      layer.weight = torch.nn.Parameter(layer.weight[sorted_indices])
      layer.bias = torch.nn.Parameter(layer.bias[sorted_indices])

    elif isinstance(layer, nn.LSTM):
      prev_layer_output_size = layer.hidden_size

      for attr in dir(layer):
        if attr.startswith('weight_ih'):
          weight = torch.stack(getattr(layer, attr).chunk(4, dim=0), dim=1)
          l1 = torch.sum(torch.abs(weight), dim=(1, 2))
          sorted_indices = torch.argsort(l1)[int(fraction*layer.hidden_size):]
          sorted_indices = torch.sort(sorted_indices).values

          bias_name = attr.replace('weight', 'bias')
          setattr(layer, attr, torch.nn.Parameter(weight[sorted_indices].view(-1, weight.shape[-1])))
          setattr(layer, bias_name, torch.nn.Parameter(getattr(layer, bias_name).view(-1, 4)[sorted_indices].view(-1)))

      for attr in dir(layer):
        if attr.startswith('weight_hh'):
          weight = torch.stack(getattr(layer, attr)[:, sorted_indices].chunk(4, dim=0), dim=1)
          l1 = torch.sum(torch.abs(weight), dim=(1, 2))
          sorted_indices = torch.argsort(l1)[int(fraction*layer.hidden_size):]
          sorted_indices = torch.sort(sorted_indices).values

          bias_name = attr.replace('weight', 'bias')
          setattr(layer, attr, torch.nn.Parameter(weight[sorted_indices].view(-1, weight.shape[-1])))
          setattr(layer, bias_name, torch.nn.Parameter(getattr(layer, bias_name).view(-1, 4)[sorted_indices].view(-1)))

      layer.hidden_size = len(sorted_indices)

    elif isinstance(layer, nn.Linear):
      l1 = torch.sum(torch.abs(layer.weight), dim=1)
      sorted_indices = torch.argsort(l1)[int(fraction*layer.in_features):]
      sorted_indices = torch.sort(sorted_indices).values

      layer.weight = torch.nn.Parameter(layer.weight[sorted_indices])

      layer.out_features = len(sorted_indices)
