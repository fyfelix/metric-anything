"""
Evaluation metrics migrated from the benchmark reference implementation.
Supports reduction='mean' and reduction='none'.
"""

import torch


def abs_relative_difference(output, target, valid_mask=None, reduction="mean"):
    actual_output = output
    actual_target = target
    abs_relative_diff = torch.abs(actual_output - actual_target) / actual_target
    if valid_mask is not None:
        abs_relative_diff[~valid_mask] = 0
        n = valid_mask.sum((-1, -2))
    else:
        n = output.shape[-1] * output.shape[-2]

    abs_relative_diff = torch.sum(abs_relative_diff, (-1, -2)) / n

    if reduction == "mean":
        return abs_relative_diff.mean()
    if reduction == "none":
        return abs_relative_diff
    raise ValueError(f"Invalid reduction: {reduction}")


def rmse_linear(output, target, valid_mask=None, reduction="mean"):
    actual_output = output
    actual_target = target
    diff = actual_output - actual_target
    if valid_mask is not None:
        diff[~valid_mask] = 0
        n = valid_mask.sum((-1, -2))
    else:
        n = output.shape[-1] * output.shape[-2]
    diff2 = torch.pow(diff, 2)
    mse = torch.sum(diff2, (-1, -2)) / n
    rmse = torch.sqrt(mse)

    if reduction == "mean":
        return rmse.mean()
    if reduction == "none":
        return rmse
    raise ValueError(f"Invalid reduction: {reduction}")


def mae_linear(output, target, valid_mask=None, reduction="mean"):
    actual_output = output
    actual_target = target
    diff = actual_output - actual_target
    if valid_mask is not None:
        diff[~valid_mask] = 0
        n = valid_mask.sum((-1, -2))
    else:
        n = output.shape[-1] * output.shape[-2]
    abs_diff = torch.abs(diff)
    mae = torch.sum(abs_diff, (-1, -2)) / n

    if reduction == "mean":
        return mae.mean()
    if reduction == "none":
        return mae
    raise ValueError(f"Invalid reduction: {reduction}")


def threshold_percentage(output, target, threshold_val, valid_mask=None, reduction="mean"):
    d1 = output / target
    d2 = target / output
    max_d1_d2 = torch.max(d1, d2)
    bit_mat = (max_d1_d2 < threshold_val).float()

    if valid_mask is not None:
        bit_mat[~valid_mask] = 0
        n = valid_mask.sum((-1, -2))
    else:
        n = output.shape[-1] * output.shape[-2]
    count_mat = torch.sum(bit_mat, (-1, -2))
    threshold_mat = count_mat / n

    if reduction == "mean":
        return threshold_mat.mean()
    if reduction == "none":
        return threshold_mat
    raise ValueError(f"Invalid reduction: {reduction}")


def delta1_acc(pred, gt, valid_mask, reduction="mean"):
    return threshold_percentage(pred, gt, 1.25, valid_mask, reduction=reduction)


def delta2_acc(pred, gt, valid_mask, reduction="mean"):
    return threshold_percentage(pred, gt, 1.25**2, valid_mask, reduction=reduction)


def delta3_acc(pred, gt, valid_mask, reduction="mean"):
    return threshold_percentage(pred, gt, 1.25**3, valid_mask, reduction=reduction)


def delta4_acc_105(pred, gt, valid_mask, reduction="mean"):
    return threshold_percentage(pred, gt, 1.05, valid_mask, reduction=reduction)


def delta5_acc110(pred, gt, valid_mask, reduction="mean"):
    return threshold_percentage(pred, gt, 1.10, valid_mask, reduction=reduction)


def delta6_acc103(pred, gt, valid_mask, reduction="mean"):
    return threshold_percentage(pred, gt, 1.03, valid_mask, reduction=reduction)
