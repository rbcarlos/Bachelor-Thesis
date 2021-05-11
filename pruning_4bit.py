# IMPORTS
import argparse
from finn.util.basic import make_build_dir
import numpy as np
from collections import OrderedDict

import onnx
from finn.util.test import get_test_model_trained
import brevitas.onnx as bo
from finn.core.modelwrapper import ModelWrapper
from finn.transformation.infer_shapes import InferShapes
from finn.transformation.fold_constants import FoldConstants
from finn.transformation.general import GiveReadableTensorNames, GiveUniqueNodeNames, RemoveStaticGraphInputs
import time
import torch
import torch.nn.utils.prune as prune
from brevitas.nn import QuantConv2d, QuantLinear


from dependencies import value

from brevitas.inject import ExtendedInjector
from brevitas.quant.solver import WeightQuantSolver, ActQuantSolver
from brevitas.core.bit_width import BitWidthImplType
from brevitas.core.quant import QuantType
from brevitas.core.restrict_val import RestrictValueType, FloatToIntImplType
from brevitas.core.scaling import ScalingImplType
from brevitas.core.zero_point import ZeroZeroPoint
from torch.nn import Module, ModuleList, BatchNorm2d, MaxPool2d, BatchNorm1d

from brevitas.nn import QuantConv2d, QuantIdentity, QuantLinear
from brevitas.core.restrict_val import RestrictValueType
import torch.nn as nn
import torch.nn.init as init

import torchvision
import torchvision.transforms as transforms

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 pruning')
parser.add_argument("--model", default="./experiments", help="Path to the pretrained model")
parser.add_argument('--epochs', default=50, type=int, metavar='N', help='Number of finetuning epochs')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float, metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--norm-order', default=1, type=int, help='Order of the vector norm')
parser.add_argument('--simd-list', default="", type=str, help='List of SIMDs for FCLayers')
parser.add_argument('--max-sparsity', default=0.9, type=float, help='Order of the vector norm')

args = parser.parse_args()

# DEFINITIONS
class CommonQuant(ExtendedInjector):
    bit_width_impl_type = BitWidthImplType.CONST
    scaling_impl_type = ScalingImplType.CONST
    restrict_scaling_type = RestrictValueType.FP
    zero_point_impl = ZeroZeroPoint
    float_to_int_impl_type = FloatToIntImplType.ROUND
    scaling_per_output_channel = False
    narrow_range = True
    signed = True

    @value
    def quant_type(bit_width):
        if bit_width is None:
            return QuantType.FP
        elif bit_width == 1:
            return QuantType.BINARY
        else:
            return QuantType.INT


class CommonWeightQuant(CommonQuant, WeightQuantSolver):
    scaling_const = 1.0


class CommonActQuant(CommonQuant, ActQuantSolver):
    min_val = -1.0
    max_val = 1.0

class TensorNorm(nn.Module):
    def __init__(self, eps=1e-4, momentum=0.1):
        super().__init__()

        self.eps = eps
        self.momentum = momentum
        self.weight = nn.Parameter(torch.rand(1))
        self.bias = nn.Parameter(torch.rand(1))
        self.register_buffer('running_mean', torch.zeros(1))
        self.register_buffer('running_var', torch.ones(1))
        self.reset_running_stats()

    def reset_running_stats(self):
        self.running_mean.zero_()
        self.running_var.fill_(1)
        init.ones_(self.weight)
        init.zeros_(self.bias)

    def forward(self, x):
        if self.training:
            mean = x.mean()
            unbias_var = x.var(unbiased=True)
            biased_var = x.var(unbiased=False)
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.detach()
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * unbias_var.detach()
            inv_std = 1 / (biased_var + self.eps).pow(0.5)
            return (x - mean) * inv_std * self.weight + self.bias
        else:
            return ((x - self.running_mean) / (self.running_var + self.eps).pow(0.5)) * self.weight + self.bias

CNV_OUT_CH_POOL = [(64, False), (64, True), (128, False), (128, True), (256, False), (256, False)]
INTERMEDIATE_FC_FEATURES = [(256, 512), (512, 512)]
LAST_FC_IN_FEATURES = 512
LAST_FC_PER_OUT_CH_SCALING = False
POOL_SIZE = 2
KERNEL_SIZE = 3


class CNV(Module):

    def __init__(self, num_classes, weight_bit_width, act_bit_width, in_bit_width, in_ch):
        super(CNV, self).__init__()

        self.conv_features = ModuleList()
        self.linear_features = ModuleList()

        self.conv_features.append(QuantIdentity( # for Q1.7 input format
            act_quant=CommonActQuant,
            bit_width=in_bit_width,
            min_val=- 1.0,
            max_val=1.0 - 2.0 ** (-7),
            narrow_range=False,
            restrict_scaling_type=RestrictValueType.POWER_OF_TWO))

        for out_ch, is_pool_enabled in CNV_OUT_CH_POOL:
            self.conv_features.append(QuantConv2d(
                kernel_size=KERNEL_SIZE,
                in_channels=in_ch,
                out_channels=out_ch,
                bias=False,
                weight_quant=CommonWeightQuant,
                weight_bit_width=weight_bit_width))
            in_ch = out_ch
            self.conv_features.append(BatchNorm2d(in_ch, eps=1e-4))
            self.conv_features.append(QuantIdentity(
                act_quant=CommonActQuant,
                bit_width=act_bit_width))
            if is_pool_enabled:
                self.conv_features.append(MaxPool2d(kernel_size=2))

        for in_features, out_features in INTERMEDIATE_FC_FEATURES:
            self.linear_features.append(QuantLinear(
                in_features=in_features,
                out_features=out_features,
                bias=False,
                weight_quant=CommonWeightQuant,
                weight_bit_width=weight_bit_width))
            self.linear_features.append(BatchNorm1d(out_features, eps=1e-4))
            self.linear_features.append(QuantIdentity(
                act_quant=CommonActQuant,
                bit_width=act_bit_width))

        self.linear_features.append(QuantLinear(
            in_features=LAST_FC_IN_FEATURES,
            out_features=num_classes,
            bias=False,
            weight_quant=CommonWeightQuant,
            weight_bit_width=weight_bit_width))
        self.linear_features.append(TensorNorm())
        
        for m in self.modules():
          if isinstance(m, QuantConv2d) or isinstance(m, QuantLinear):
            torch.nn.init.uniform_(m.weight.data, -1, 1)


    def clip_weights(self, min_val, max_val):
        for mod in self.conv_features:
            if isinstance(mod, QuantConv2d):
                mod.weight.data.clamp_(min_val, max_val)
        for mod in self.linear_features:
            if isinstance(mod, QuantLinear):
                mod.weight.data.clamp_(min_val, max_val)

    def forward(self, x):
        x = 2.0 * x - torch.tensor([1.0], device=x.device)
        for mod in self.conv_features:
            x = mod(x)
        x = x.view(x.shape[0], -1)
        for mod in self.linear_features:
            x = mod(x)
        return x


def cnv(weight_bit_width, act_bit_width, in_bit_width):
    num_classes = 10
    in_channels = 3
    net = CNV(weight_bit_width=weight_bit_width,
              act_bit_width=act_bit_width,
              in_bit_width=in_bit_width,
              num_classes=num_classes,
              in_ch=in_channels)
    return net

# LOAD DATA
transform = transforms.Compose(
    [transforms.ToTensor()])

train_transforms_list = [transforms.RandomCrop(32, padding=4),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor()]
transform_train = transforms.Compose(train_transforms_list)

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                          shuffle=True, num_workers=1)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=128,
                                         shuffle=False, num_workers=1)

model = cnv(4,4,8)

package = torch.load(args.model, map_location='cpu')
model_state_dict = package['state_dict']
model.load_state_dict(model_state_dict, strict=False)


# LOAD MODEL
#model = get_test_model_trained("CNV", 2, 2)

import torch.optim as optim
optimizer = optim.Adam(model.parameters(), lr=args.lr)
device = 'cuda:0'
criterion = nn.CrossEntropyLoss().to(device)

# TRAINING AND TESTING
def test():
  model.to(device)
  model.eval()
  criterion.eval()

  prec1_global = []

  for i, data in enumerate(testloader):

      #print("Batch", i+1)

      (input, target) = data

      input = input.to(device, non_blocking=True)
      target = target.to(device, non_blocking=True)
      
      # compute output
      output = model(input)

      #compute loss
      loss = criterion(output, target)

      pred = output.data.argmax(1, keepdim=True)
      correct = pred.eq(target.data.view_as(pred)).sum()
      prec1 = 100. * correct.float() / input.size(0)

      #print("Acc1:", prec1)

      prec1_global.append(prec1)
  print("Global top1 val acc:", np.mean([x.item() for x in prec1_global]))
  return np.mean([x.item() for x in prec1_global])

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].flatten().float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def train(epochs=5, filename="best.tar"):
  best_acc = 0
  accs = []
  for epoch in range(epochs):
    # Set to training mode
    model.to(device)
    model.train()
    criterion.train()

    for i, data in enumerate(trainloader):
        (input, target) = data
        input = input.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # Training batch starts
        output = model(input)
        loss = criterion(output, target)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.clip_weights(-1,1)

        if i%10==0:
            prec1, prec5 = accuracy(output.detach(), target, topk=(1, 5))
            print(f"Epoch {epoch+1}, batch {i+1}: top1 acc = {prec1}")

    if epoch%40==0:
      optimizer.param_groups[0]['lr'] *= 0.5

    val_acc = test()
    accs.append(val_acc)
    if val_acc > best_acc:
      print(str(val_acc) + " is higher than " + str(best_acc) + ", saving")
      best_acc = val_acc
      torch.save({
            'state_dict': model.state_dict(),
            'optim_dict': optimizer.state_dict(),
            'best_val_acc': best_acc,
        }, filename)
  return accs

# PRUNING DEFINITIONS
def make_weights_during_training(model_state_dict):
  for orig_weight_key, mask_key in [('conv_features.1.weight_orig', 'conv_features.1.weight_mask'), ('conv_features.4.weight_orig', 'conv_features.4.weight_mask'), ('conv_features.8.weight_orig', 'conv_features.8.weight_mask') , ('conv_features.11.weight_orig', 'conv_features.11.weight_mask'), ('conv_features.15.weight_orig', 'conv_features.15.weight_mask'), ('conv_features.18.weight_orig', 'conv_features.18.weight_mask')]:
    orig_weight = model_state_dict[orig_weight_key]
    #print(model_state_dict[mask_key].shape)
    export_mask = model_state_dict[mask_key][0]
    #print(export_mask.tolist())
    #export_mask_list.append(export_mask.tolist())
    mask = model_state_dict[mask_key].bool()
    orig_weight[~mask] = 0
    weight = orig_weight
    weight_key = '.'.join(orig_weight_key.split(".")[:2])+ ".weight"
    model_state_dict = OrderedDict([(weight_key, weight) if k == orig_weight_key else (k, v) for k, v in model_state_dict.items()])
    model_state_dict.pop(mask_key)
  return model_state_dict

from torch.nn.utils.prune import BasePruningMethod
from torch.nn.utils.prune import _validate_pruning_amount_init, _validate_structured_pruning

class PruneSIMD(BasePruningMethod):
    r"""Prune entire (currently unpruned) channels in a tensor at random.
    Args:
        amount (int or float): quantity of parameters to prune.
            If ``float``, should be between 0.0 and 1.0 and represent the
            fraction of parameters to prune. If ``int``, it represents the 
            absolute number of parameters to prune.
        dim (int, optional): index of the dim along which we define
            channels to prune. Default: -1.
    """

    PRUNING_TYPE = "unstructured"

    def __init__(self, amount, SIMD, n):
        self.amount = amount
        self.SIMD = SIMD
        self.n = n

    def compute_mask(self, t, default_mask):
        r"""Computes and returns a mask for the input tensor ``t``.
        Starting from a base ``default_mask`` (which should be a mask of ones
        if the tensor has not been pruned yet), generate a random mask to 
        apply on top of the ``default_mask`` by randomly zeroing out channels
        along the specified dim of the tensor.
        Args:
            t (torch.Tensor): tensor representing the parameter to prune
            default_mask (torch.Tensor): Base mask from previous pruning 
                iterations, that need to be respected after the new mask is 
                applied. Same dims as ``t``.
        Returns:
            mask (torch.Tensor): mask to apply to ``t``, of same dims as ``t``
        Raises:
            IndexError: if ``self.dim >= len(t.shape)``
        """
        
        n_channels = torch.flatten(t, start_dim=1).shape[1]

        if not (n_channels % self.SIMD == 0):
            raise ValueError(f"n_channels={n_channels} must be divisible by SIMD={self.SIMD}")

        new_shape = n_channels // self.SIMD
        params_to_keep = new_shape - int(np.round(new_shape * self.amount))
        if params_to_keep > self.SIMD :
            params_to_keep += self.SIMD
            params_to_keep -= params_to_keep % self.SIMD
        
        n_SIMD_channels = n_channels // self.SIMD
        params_to_prune = n_SIMD_channels - params_to_keep

        if params_to_prune == 0:
            return default_mask

        t = t.permute(0, 2, 3, 1)
        
        flat_t = torch.flatten(t, start_dim=1)
        
        norms_of_blocks = []
        for i in range(n_SIMD_channels):
            block = flat_t[:, i*self.SIMD : i*self.SIMD + self.SIMD]
            norm = torch.norm(block, p=self.n)
            norms_of_blocks.append(norm)

        norms_of_blocks = torch.tensor(norms_of_blocks)
        threshold = torch.kthvalue(norms_of_blocks, k=params_to_prune).values
        
        mask = torch.zeros_like(t)
        mask_flat = torch.flatten(mask, start_dim=1)
        for i in range(n_SIMD_channels):
          if norms_of_blocks[i] > threshold:
            mask_flat[:, i*self.SIMD : i*self.SIMD + self.SIMD] = 1    
        # reshape the mask to from NHWC to NCHW
        mask = mask_flat.view(t.shape)
        mask = mask.permute(0, 3, 1, 2)
        mask *= default_mask.to(dtype=mask.dtype)
        return mask
        
    @classmethod
    def apply(cls, module, name, amount, SIMD, n):
        r"""Adds the forward pre-hook that enables pruning on the fly and
        the reparametrization of a tensor in terms of the original tensor
        and the pruning mask.
        Args:
            module (nn.Module): module containing the tensor to prune
            name (str): parameter name within ``module`` on which pruning
                will act.
            amount (int or float): quantity of parameters to prune.
                If ``float``, should be between 0.0 and 1.0 and represent the
                fraction of parameters to prune. If ``int``, it represents the 
                absolute number of parameters to prune.
            dim (int, optional): index of the dim along which we define
                channels to prune. Default: -1.
        """
        return super(PruneSIMD, cls).apply(
            module, name, SIMD=SIMD, amount=amount, n=n
        )

def prune_simd(increment = 0.1, start_sparsity=0.5, max_sparsity = 0.7, finetune_epochs=5):
  i = 1
  test_acc = []
  sparsity = []

  while True:
    #print("shape",model.conv_features[1].weight.shape)
    sparsity_before = 100. * float(
                torch.sum(model.conv_features[1].weight == 0)
                + torch.sum(model.conv_features[4].weight == 0)
                + torch.sum(model.conv_features[8].weight == 0)
                + torch.sum(model.conv_features[11].weight == 0)
                + torch.sum(model.conv_features[15].weight == 0)
                + torch.sum(model.conv_features[18].weight == 0)
            ) / float(
                model.conv_features[1].weight.nelement()
                + model.conv_features[4].weight.nelement()
                + model.conv_features[8].weight.nelement()
                + model.conv_features[11].weight.nelement()
                + model.conv_features[15].weight.nelement()
                + model.conv_features[18].weight.nelement()
            )
    print("Global sparsity before pruning: {:.2f}%".format(sparsity_before))
    filename = "best_4bit_" + str(int(sparsity_before)) + "_pruned_" + str(args.max_sparsity) + ".tar"

    sparsity.append(sparsity_before)
    
    if i!=1:
      package = torch.load(filename, map_location='cpu')
      model_state_dict = package['state_dict']
      model.load_state_dict(make_weights_during_training(model_state_dict), strict=True)
      print("Loaded model with acc", str(package["best_val_acc"]))
      if 'optim_dict' in package.keys():
        optimizer.load_state_dict(package['optim_dict'])

    test_acc.append(test())

    parameters_to_prune = [
        (model.conv_features[1], 'weight'),
        (model.conv_features[4], 'weight'),
        (model.conv_features[8], 'weight'),
        (model.conv_features[11], 'weight'),
        (model.conv_features[15], 'weight'),
        (model.conv_features[18], 'weight'),
    ]

    SIMD = [int(x) for x in args.simd_list.split(",")]

    if i==1:
      amount = start_sparsity + increment
    else:
      #amount = increment / (1-(sparsity_before /100))
      amount = (sparsity_before/100) + increment

    for j, (layer, param) in enumerate(parameters_to_prune):
      PruneSIMD.apply(layer, name=param, amount=amount, n=args.norm_order, SIMD=SIMD[j])

    sparsity_after = 100. * float(
              torch.sum(model.conv_features[1].weight == 0)
              + torch.sum(model.conv_features[4].weight == 0)
              + torch.sum(model.conv_features[8].weight == 0)
              + torch.sum(model.conv_features[11].weight == 0)
              + torch.sum(model.conv_features[15].weight == 0)
              + torch.sum(model.conv_features[18].weight == 0)
          ) / float(
              model.conv_features[1].weight.nelement()
              + model.conv_features[4].weight.nelement()
              + model.conv_features[8].weight.nelement()
              + model.conv_features[11].weight.nelement()
              + model.conv_features[15].weight.nelement()
              + model.conv_features[18].weight.nelement()
          )
    filename = "best_4bit_" + str(int(sparsity_after)) + "_pruned_" + str(args.max_sparsity) + ".tar"
    print("Global sparsity after pruning: {:.2f}%".format(sparsity_after))
    print("Testing before finetuning")
    test()
    print("Finetune")
    modules = [model.conv_features[1],model.conv_features[4],model.conv_features[8],model.conv_features[11],model.conv_features[15],model.conv_features[18]]
    #print("shape before", model.conv_features[1].weight.shape)
    epoch_acc = train(finetune_epochs, filename)
    print(f"Sparsity {sparsity_after}%: {epoch_acc}")
    #print("shape after", model.conv_features[1].weight.shape)
    for j, module in enumerate(modules):
      prune.remove(module, 'weight')
    if sparsity_after > max_sparsity*100:
      test_acc.append(test())
      sparsity.append(sparsity_after)
      break
    i+=1
  print("Test acc", test_acc)
  print("Sparsity",sparsity)
  return test_acc, sparsity

# PRUNE
sparsity, val_acc = prune_simd(start_sparsity=0.0, increment=0.15, max_sparsity=args.max_sparsity, finetune_epochs=args.epochs)