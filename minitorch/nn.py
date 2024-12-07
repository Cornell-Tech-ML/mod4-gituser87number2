from typing import Tuple

from . import operators
from .autodiff import Context
from .fast_ops import FastOps
from .tensor import Tensor
from .tensor_functions import Function, rand

# List of functions in this file:
# - avgpool2d: Tiled average pooling 2D
# - argmax: Compute the argmax as a 1-hot tensor
# - Max: New Function for max operator
# - max: Apply max reduction
# - softmax: Compute the softmax as a tensor
# - logsoftmax: Compute the log of the softmax as a tensor - See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
# - maxpool2d: Tiled max pooling 2D
# - dropout: Dropout positions based on random noise, include an argument to turn off


def tile(input: Tensor, kernel: Tuple[int, int]) -> Tuple[Tensor, int, int]:
    """Reshape an image tensor for 2D pooling

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width x (kernel_height * kernel_width) as well as the new_height and new_width value.

    """
    batch, channel, height, width = input.shape
    kh, kw = kernel
    assert height % kh == 0
    assert width % kw == 0

    new_height = height // kh
    new_width = width // kw

    # Reshape input to split height and width into kernel sizes
    input = input.contiguous().view(batch, channel, new_height, kh, new_width, kw)

    # Swap kh and new_width to move kh, kw to the last dimension
    input = input.permute(0, 1, 2, 4, 3, 5).contiguous()

    # Reshape to combine the height and width kernel
    input = input.view(batch, channel, new_height, new_width, kh * kw)

    return input, new_height, new_width


def avgpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Apply average pooling to input tensor given kernel size

    Args:
    ----
        input: Tensor of size batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width

    """
    batch, channel, _, _ = input.shape
    tiled, new_height, new_width = tile(input, kernel)

    # Calculates mean over dim 4 of tiled tensor
    return tiled.mean(dim=4).view(batch, channel, new_height, new_width)


max_reduce = FastOps.reduce(operators.max, -1e9)
# uses fast ops to reduce the tensor by the max value
# -1e9 is used to prevent overflow errors
# used in argmax func and max class


def argmax(input: Tensor, dim: int) -> Tensor:
    """Compute the argmax as a 1-hot tensor

    Args:
    ----
        input: batch x channel x height x width
        dim: dimension to reduce over

    Returns:
    -------
        Tensor of size batch x channel x height x width where argmax is 1, else 0

    """
    return input == max_reduce(input, dim)


class Max(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, dim: Tensor) -> Tensor:
        """Forward pass for Max is max reduction"""
        dim_val = int(dim.item())
        ctx.save_for_backward(input, dim_val)
        return max_reduce(input, dim_val)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Backward pass for Max is argmax
        Derivative of max is 1 if it is the max value, else 0
        Acts as a gate for the gradient
        """
        input, dim = ctx.saved_values
        return grad_output * argmax(input, dim), 0.0


def max(input: Tensor, dim: int) -> Tensor:
    """Apply max reduction to the input tensor given dimension."""
    return Max.apply(input, input._ensure_tensor(dim))


def softmax(input: Tensor, dim: int) -> Tensor:
    """Compute the softmax as a tensor

    Args:
    ----
        input: batch x channel x height x width
        dim: dimension to apply softmax over

    Returns:
    -------
        Tensor of size batch x channel x height x width with softmax applied to dim

    """
    input_exp = input.exp()
    sum_exp = input_exp.sum(dim)  # compute softmax
    return input_exp / sum_exp


def logsoftmax(input: Tensor, dim: int) -> Tensor:
    """Compute the log of the softmax as a tensor

    Args:
    ----
        input: batch x channel x height x width
        dim: dimension to apply logsoftmax over

    Returns:
    -------
        Tensor of size batch x channel x height x width with logsoftmax applied to dim

    """
    x_max = max(input, dim)
    input_exp = (input - x_max).exp()
    sum_exp = input_exp.sum(dim)
    log_sum_exp = sum_exp.log() + x_max
    return input - log_sum_exp


def maxpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Apply max pooling to the input tensor given kernel size.

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width

    """
    batch, channel, _, _ = input.shape
    tiled, new_height, new_width = tile(input, kernel)

    # Max over the last dimension
    return max(tiled, dim=4).view(batch, channel, new_height, new_width)


def dropout(input: Tensor, p: float, ignore: bool = False) -> Tensor:
    """Dropout positions based on random noise, option to turn off

    Args:
    ----
        input: batch x channel x height x width
        p: probability of dropout
        ignore: option to ignore dropout

    Returns:
    -------
        Tensor of same size as input with dropout applied

    """
    if ignore == 1:
        return input

    # Mask of 1 and 0 values based on probability p
    # If value is greater than p value is kept, else 0
    random_mask = rand(input.shape) > p

    return input * random_mask
