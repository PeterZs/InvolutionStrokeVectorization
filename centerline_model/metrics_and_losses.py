import torch
import cv2
import numpy as np
from typing import Dict, Literal

import torch.distributed as dist
import torch.nn.functional as F
from torchmetrics import Metric

def gumbel_noise(shape, device, eps=1e-20):
    u = torch.rand(shape, device=device)
    return -torch.log(-torch.log(u + eps) + eps)

def distance_transform(img: torch.Tensor, mode: Literal['inside', 'outside'] = 'outside', debug: bool = False):
    """
    Compute the distance transform of a 2d torch.Tensor with values in [0, 1].
    """
    if len(img.shape) != 2:
        raise ValueError(f"Input tensor must be 2D, but is {img.shape}")

    img_np = img.detach().cpu().numpy()

   
    if mode == 'outside':
        binary = (img_np != 0).astype(np.uint8)
        source = 1 - binary
    elif mode == 'inside':
        source = (img_np !=0).astype(np.uint8)
    dist = cv2.distanceTransform(source, distanceType=cv2.DIST_L2, maskSize=5)
    dist_tensor = torch.from_numpy(dist).to(dtype=img.dtype, device=img.device)

    if debug:
        dist[dist>25] = 25
        dist_normalized = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return dist_tensor, dist_normalized
    
    return dist_tensor, None

def get_outside_distance_transforms(imgs: torch.Tensor):
    dt = []
    for img in imgs:
        curr_dt, debug = distance_transform(img.squeeze(), 'outside', debug=False)
        #cv2.imwrite("distance_debug.png", debug)
        dt.append(curr_dt)
    return torch.stack(dt)

def gradient_x(img): return img[:, :, :, 1:] - img[:, :, :, :-1]
def gradient_y(img): return img[:, :, 1:, :] - img[:, :, :-1, :]

def gradloss(preds: torch.Tensor, targets: torch.Tensor):
    grad_x_pred = gradient_x(preds)
    grad_y_pred = gradient_y(preds)

    grad_x_gt = gradient_x(targets)
    grad_y_gt = gradient_y(targets)

    L_grad = ( (grad_x_pred - grad_x_gt).abs().mean() +
            (grad_y_pred - grad_y_gt).abs().mean() )
    return L_grad


def median_blur(images: torch.Tensor) -> torch.Tensor:
    """
    Applies a 5px median blur

    Args:
        images: Tensor of shape (Batch, Channel, Height, Width)

    Returns:
        Tensor
    """
    # 1. Prepare for 5px Median Blur
    kernel_size = 5
    padding = kernel_size // 2  # Padding = 2

    # We pad the image with reflection to handle borders smoothly
    # Pad order: (left, right, top, bottom)
    x_padded = F.pad(images, (padding, padding, padding, padding), mode='reflect')

    # 2. Unfold to get sliding windows
    # We want a sliding window of 5x5. 
    # Unfolding creates a view of the tensor where local blocks are stacked.
    # Unfold dimension 2 (Height)
    unfolded = x_padded.unfold(dimension=2, size=kernel_size, step=1)
    # Unfold dimension 3 (Width)
    unfolded = unfolded.unfold(dimension=3, size=kernel_size, step=1)

    # Current shape: (B, C, H, W, kernel_size, kernel_size)
    # We flatten the kernel dimensions to find the median
    unfolded = unfolded.contiguous().view(
        images.shape[0], images.shape[1], images.shape[2], images.shape[3], -1
    )

    # 3. Compute Median
    # .median() returns a named tuple (values, indices), we select values [0]
    blurred, _ = unfolded.median(dim=-1)

    # 4. Threshold (Keep values > 0.04, set others to 0)
    # F.threshold(input, threshold, value_to_replace)
    # blurred = F.threshold(blurred, 0.04, 0.0)

    # 5. Sum everything up
    return blurred

def median_sum_loss(preds: torch.Tensor, targets: torch.Tensor)->torch.Tensor:
    blurred = median_blur(preds)
    blurred = F.threshold(blurred, 0.04, 0.0)
    return blurred.sum()

def mse_gumbel_loss(preds: torch.Tensor, targets: torch.Tensor)->torch.Tensor:
    noise =gumbel_noise(preds.shape, preds.device)
    
    return F.mse_loss(preds+noise, targets, reduction='sum')

def distanceloss(preds: torch.Tensor, targets: torch.Tensor):
    dt = get_outside_distance_transforms(targets)
    return (preds*dt).norm()

def bimodal_sharpness_loss(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Penalizes values near 0.5 to force predictions towards 0 or 1.
    `targets` is accepted for API consistency but is not used.
    """
    # Max penalty at 0.5 (0.5 * 0.5 = 0.25)
    # Min penalty at 0.0 or 1.0 (0 * 1 = 0)
    return (preds * (1.0 - preds)).mean()


def soft_dice_loss(preds: torch.Tensor, targets: torch.Tensor, mask = None, smooth: float = 1e-5) -> torch.Tensor:
    """
    Evaluates global structural overlap rather than pixel-by-pixel accuracy.
    Helps solidify shapes and prevents broad, blurry edges while allowing 
    steep anti-aliased gradients.
    """
    # Flatten tensors to compute global intersection and union
    pred_flat = preds.view(-1)
    target_flat = targets.view(-1)
    
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum()
    
    # Calculate Dice coefficient
    dice = (2.0 * intersection + smooth) / (union + smooth)
    
    # Return loss (1 - dice)
    return 1.0 - dice



def masked_l2_loss(pred_batch: torch.Tensor, gt_batch: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    # Zero out values outside the mask
    preds_masked = pred_batch * masks
    gt_masked = gt_batch * masks

    # areas: Sum of the mask (valid pixels) per batch item
    # flatten(1) turns (B, C, H, W) into (B, Features), allowing a simple sum(dim=1)
    areas = masks.flatten(1).sum(dim=1)

    # # l2s: Sum of Squared Error (SSE) per batch item
    l2s = (preds_masked - gt_masked).pow(2).flatten(1).sum(dim=1)

    # den = (torch.max((1-gt_batch).pow(2), gt_batch.pow(2))*masks).flatten(1).sum(dim=1)

    # # Return MSE per valid area (Shape: [B])
    # # clamp(min=1e-8) prevents division by zero if a mask is completely empty
    return l2s

def masked_median_area(pred_batch: torch.Tensor, gt_batch: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    
    medians = median_blur(pred_batch)
    medians_masked = medians * masks
    areas = masks.flatten(1).sum(dim=1)
    median_areas = medians_masked.flatten(1).sum()
    return median_areas/areas.clamp(min=1e-8)



def build_loss_func(losses: dict, device: torch.device):
    """
    Build a weighted sum loss function from a dictionary of loss names -> weights.

    Args:
        losses (dict): {"LossName": weight}, supported LossName: "L1", "MSE", "BCE", etc.

    Returns:
        A function loss_fn(preds, targets) -> scalar loss
    """

    # Map strings to actual PyTorch loss functions
    loss_map = {
        "L1Loss": torch.nn.L1Loss(),
        "MSELoss": torch.nn.MSELoss(reduction='mean'),
        "dtloss": distanceloss,
        "gradloss": gradloss,
        "median_sum_loss": median_sum_loss,
        "mse_gumbel_loss": mse_gumbel_loss,
        "bimodal_sharpness_loss": bimodal_sharpness_loss,
        "soft_dice_loss": soft_dice_loss,
    }

    # Filter valid losses
    active_losses = {}
    for name, weight in losses.items():
        if name in loss_map and weight > 0:
            active_losses[name] = (loss_map[name], weight)
        else:
            raise ValueError(f"Unsupported loss '{name}' or unvalid weight: '{weight}")

    # Return a function that computes weighted sum
    def loss_fn(preds, targets):
        total_loss = torch.tensor(0.0, device=device)
        components = {}
        for name, (loss_func, weight) in active_losses.items():
            curr = loss_func(preds, targets)
            components[name] = curr
            total_loss += weight * curr
        return total_loss, components

    return loss_fn


class GlobalStatsMetric(Metric):
    full_state_update = False

    def __init__(self, metric_func, smaller_is_better = True, name = None, **kwargs):
        super().__init__(**kwargs)

        # 1. Store ALL scalar scores to compute Mean/Median/Std/etc.
        # dist_reduce_fx="cat" will concatenate results from all ranks automatically.
        self.add_state("scores", default=[], dist_reduce_fx="cat")

        # 2. Local tracking for the "Worst" sample (tensors).
        # We handle this manually in Python to avoid complex TorchMetrics logic.
        self._reset_local_tracking()
        self.name = name
        if name is None:
            self.name = metric_func.__name__
        self.func= metric_func

    def _reset_local_tracking(self):
        self.worst_score = float("-inf")
        # This will hold a dictionary of tensors for the single worst sample, e.g.:
        # {'input': Tensor, 'target': Tensor, 'pred': Tensor} (all on CPU)
        self.worst_sample_data = None

    def reset(self):
        super().reset()
        self._reset_local_tracking()

    def update(self, metric_args: tuple, sample_data: Dict[str, torch.Tensor]):
        """
        metric_values: Shape (Batch,) - The score for each sample (higher is "worse" or "better" depending on your logic)
        sample_data: Dict of batches, e.g. {'img': (B,C,H,W), 'pred': (B, ...)}
        """
        # --- 1. Accumulate Scores for Global Stats ---
        # Detach and move to CPU immediately
        current_scores = self.func(*metric_args).detach().flatten().cpu()
        self.scores.append(current_scores)

        # --- 2. Update Local "Worst" Sample ---
        # Find the max score in this specific batch
        batch_max_val, batch_idx = torch.max(current_scores, dim=0)
        batch_max_val = batch_max_val.item()

        # If this batch contains a new global max (for this rank), overwrite our stored sample
        if batch_max_val > self.worst_score:
            self.worst_score = batch_max_val

            # Extract the specific slice corresponding to batch_idx and move to CPU
            # We construct a new dict containing only that single sample
            self.worst_sample_data = {}
            for k, v in sample_data.items():
                if isinstance(v[batch_idx], str):
                    self.worst_sample_data[k] = v[batch_idx]
                else:
                # v[batch_idx] extracts the specific sample
                    self.worst_sample_data[k] = v[batch_idx].detach().cpu()

    def compute(self):
        # --- 1. Compute Numeric Stats ---
        if not self.scores:
            return {}

        all_scores = torch.cat(self.scores, dim=0).float()

        results = {
            f"{self.name} : mean": torch.mean(all_scores),
            f"{self.name} : median": torch.median(all_scores),
            f"{self.name} : std": torch.std(all_scores),
            f"{self.name} : min": torch.min(all_scores),
            f"{self.name} : max": torch.max(all_scores),
        }

        # --- 2. Find Global Worst Sample (DDP Sync) ---

        # Current rank's candidate
        global_worst_score = self.worst_score
        global_worst_data = self.worst_sample_data

        if dist.is_available() and dist.is_initialized():
            # Gather candidates from all ranks
            # We package score + data into a simple object
            my_candidate = {
                "score": self.worst_score,
                "data": self.worst_sample_data
            }

            world_size = dist.get_world_size()
            candidates = [None] * world_size

            # all_gather_object handles pickling dicts/tensors automatically
            dist.all_gather_object(candidates, my_candidate)

            # Find the winner among the gathered candidates
            # Reset to find the max in the list
            global_worst_score = float("-inf")

            for cand in candidates:
                if cand["score"] > global_worst_score:
                    global_worst_score = cand["score"]
                    global_worst_data = cand["data"]

        # Add the worst sample tensors to the results
        # global_worst_data is a dict like {'img': Tensor, 'pred': Tensor}
        # results["worst_sample"] = global_worst_data

        return results, global_worst_data