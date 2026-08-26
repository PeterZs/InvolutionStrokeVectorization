import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Optional

class MockModel(nn.Module):
    def __init__(self, max_objects=1000, **kwargs):
        super().__init__()
        self.max_objects = max_objects
        
        # A trivial "dummy" layer. 
        # Instead of extracting features, we just learn a static bias for every possible 
        # pair of objects (N x N).
        # We assume the input mask might just be flattened, but for this mock, 
        # we completely ignore the pixel-level mask and image features.
        
        self.static_relation_bias = nn.Parameter(torch.randn(max_objects, max_objects))
        
        # A dummy layer to process the image so PyTorch sees "image" 
        # as part of the computation graph (useful for debugging DDP/Gradient checks)
        self.dummy_img_process = nn.Conv2d(3, 1, kernel_size=1)

    def forward(self, img, seg_mask, relation_mask, n_objects_int):
        """
        img: [B, 3, H, W] - RGB Image
        seg_mask: [B, H, W] - Segmentation indices (0 to N-1)
        relation_mask: [B, N, N] - The constraint matrix R (1=valid, 0=invalid)
        """
        assert img.shape[1] == 3
        B = img.shape[0]
        K = self.max_objects

        # 1. Trivial Image Processing (Wrong, but ensures gradients flow)
        # Reduce image to a single scalar per batch just to involve it
        dummy_feat = self.dummy_img_process(img).mean(dim=[1, 2, 3]) # [B]
        # 2. Generate Logits
        # We take our learned static bias and expand it to the batch size.
        # We add the dummy image scalar to prove data flows from input to output.
        # Slice to current N (in case max_objects > N)
        logits = self.static_relation_bias.unsqueeze(0).expand(B, -1, -1)
        logits = logits + dummy_feat.view(B, 1, 1)
        assert logits.shape == (B, K, K)

        # 3. Apply the Relation Mask (R)
        # Where R is 0 (False), set logits to Negative Infinity.
        # This ensures probability becomes exactly 0 after Softmax.
        bool_mask = relation_mask.bool()
        logits = logits.masked_fill(~bool_mask, float('-inf'))

        # 4. Distribute Mass (Softmax)
        # Apply softmax over the last dimension (columns).
        # Row i sums to 1 (distributed among valid columns j).
        attn_weights = F.softmax(logits, dim=-1)

        # 5. Handle Edge Cases (Rows with all zeros)
        # If an object has NO valid relations, softmax(-inf) = NaN.
        # Convert NaNs to 0.0
        attn_weights = torch.nan_to_num(attn_weights, 0.0)

        return attn_weights

def batched_sinkhorn(A, n_iters=10, epsilon=1e-8):
    """
    Performs Sinkhorn normalization on a batch of matrices.
    
    A: (B, N, M) batch of matrices (non-negative)
    n_iters: number of normalization iterations
    """
    B, N, M = A.shape
    X = A.clone()
    
    for _ in range(n_iters):
        # normalize rows
        X = X / (X.sum(dim=2, keepdim=True) + epsilon)
        # normalize columns
        X = X / (X.sum(dim=1, keepdim=True) + epsilon)
    
    return X



def standardize_logits(M: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Standardizes log-potentials (logits) per batch item to have roughly 
    zero mean and unit variance, ignoring -inf entries.
    
    This solves the "Cold Transport" problem where large logit magnitudes 
    cause Sinkhorn to freeze or fail to converge.

    Args:
        M (torch.Tensor): Logits of shape (B, N, N). 
                          Invalid entries/padding must be -inf.
        eps (float): Small epsilon for numerical stability.

    Returns:
        torch.Tensor: Standardized logits (B, N, N). -inf entries remain -inf.
    """
    # 1. Create a mask for finite values
    # We use > -inf instead of != -inf to handle potential NaNs safely if any exist
    mask = (M > -float('inf'))
    
    # 2. Create a clean tensor for statistics calculation
    # Replace -inf with 0.0 so they don't affect sums
    M_clean = torch.where(mask, M, torch.zeros_like(M))
    
    # 3. Compute count of valid entries per batch
    # Shape: (B, 1, 1) for broadcasting
    counts = mask.sum(dim=(1, 2), keepdim=True).float()
    counts = counts.clamp(min=1.0)
    
    # 4. Compute Mean
    # Shape: (B, 1, 1)
    mean = M_clean.sum(dim=(1, 2), keepdim=True) / counts
    
    # 5. Compute Variance / Std
    # We calculate variance manually: sum((x - mean)^2) / N
    # We must mask the subtraction so padding doesn't contribute (0 - mean)^2
    M_centered = torch.where(mask, M - mean, torch.zeros_like(M))
    var = (M_centered ** 2).sum(dim=(1, 2), keepdim=True) / counts
    std = torch.sqrt(var + eps)
    
    # 6. Apply Standardization
    # (M - mean) / std
    # PyTorch handles -inf correctly here: (-inf - mean) / std = -inf
    M_standardized = (M - mean) / std
    
    return M_standardized

def symmetric_rand(shape, device=None):
    R = torch.rand(shape, device=device)
    A = torch.triu(R)
    A = A + A.transpose(-1, -2) - torch.diag_embed(torch.diagonal(A, dim1=-2, dim2=-1))
    return A

def symmetric_gumbel(shape, device=None, eps=1e-20):
    u = symmetric_rand(shape, device=device)
    return -torch.log(-torch.log(u + eps) + eps)

def log_sinkhorn_iterations2(M: torch.Tensor, num_iters: int = 20, epsilon: float = 1.0):
    """
    Stable, Differentiable Batched Log-Sinkhorn.
    """
    # 1. Scale
    M_scaled = M / epsilon

    # 2. Precompute Masks
    # Shape: (B, N)
    # A row is valid if it has at least one finite entry
    valid_row_mask = (M > -float('inf')).any(dim=2)
    valid_col_mask = (M > -float('inf')).any(dim=1)
    
    # We need to broadcast these to (B, N, N) for the updates below
    # Row mask needs to shield dim 1 (rows) when summing over dim 2
    row_mask_expanded = valid_row_mask.unsqueeze(2) # (B, N, 1)
    # Col mask needs to shield dim 2 (cols) when summing over dim 1
    col_mask_expanded = valid_col_mask.unsqueeze(1) # (B, 1, N)

    # 3. Initialize dual variables
    u = torch.zeros_like(M[:, :, 0])
    v = torch.zeros_like(M[:, 0, :])

    for _ in range(num_iters):
        # --- Update u (Row Normalization) ---
        
        # 1. Add v to M
        M_plus_v = M_scaled + v.unsqueeze(1)
        
        # 2. CRITICAL FIX: Mask -inf BEFORE logsumexp
        # If a row is invalid, the sum is exp(-inf) = 0. Div by 0 in grad.
        # We temporarily fill invalid rows with 0 so logsumexp is safe.
        # The result of this row will be discarded in step 4 anyway.
        m_plus_v_safe = M_plus_v.masked_fill(~row_mask_expanded, 0.0)
        
        # 3. LogSumExp
        row_lse = torch.logsumexp(m_plus_v_safe, dim=2)
        
        # 4. Mask the Output
        # For valid rows: keep the calculation.
        # For invalid rows: force to 0 (so u becomes 0).
        row_lse = torch.where(valid_row_mask, row_lse, torch.zeros_like(row_lse))
        
        u = -row_lse

        # --- Update v (Column Normalization) ---
        
        M_plus_u = M_scaled + u.unsqueeze(2)
        
        # CRITICAL FIX: Mask -inf BEFORE logsumexp
        m_plus_u_safe = M_plus_u.masked_fill(~col_mask_expanded, 0.0)
        
        col_lse = torch.logsumexp(m_plus_u_safe, dim=1)
        
        col_lse = torch.where(valid_col_mask, col_lse, torch.zeros_like(col_lse))
        
        v = -col_lse

    # 4. Compute Final P
    log_P = u.unsqueeze(2) + M_scaled + v.unsqueeze(1)
    P = torch.exp(log_P)

    # 5. Final Masking
    mask = (M > -float('inf'))
    P = P * mask.float()

    return P


def gumbel_sinkhorn(preds: torch.Tensor, tau = 1, num_iters=20):
    g = symmetric_gumbel(preds.shape, device=preds.device)
    return log_sinkhorn_iterations2(preds+g, num_iters=num_iters, epsilon=tau)

def scatter_2d(features: torch.Tensor, seg_mask: torch.Tensor, K: int, check_bounds: bool = False):
    # 2. Native Torch Scatter Mean
    B, C, H, W = features.shape
    assert seg_mask.shape == (B, H, W)
    
    # if check_bounds: 
    #     mn, mx = seg_mask.min(), seg_mask.max()
    #     #print("bounds", mn, mx)
    #     if mn < 0:
    #         raise RuntimeError("seg index <0")
    #     if mx >= K:
    #         raise RuntimeError("seg index >= K")
    
    # Flatten spatial dimensions
    flat_feat = features.view(B, C, -1)     # [B, C, H*W]
    flat_mask = seg_mask.view(B, -1)        # [B, H*W]
    
    assert flat_feat.shape == (B, C, H*W)
    assert flat_mask.shape == (B, H*W)
    
    # Expand mask to match feature channels (required for scatter)
    # Shape: [B, C, H*W]
    index = flat_mask.unsqueeze(1).expand(B, C, -1)
    assert index.shape == (B, C, H*W)
    
    # Allocate output buffer
    obj_embeddings = torch.zeros(B, C, K, device=features.device, dtype=features.dtype)
    
    
    # We use scatter_reduce_ to average features into the K slots
    # index [B, C, H*W], features [B, C, H*W] -> [B, C, K]
    # dim=2 means we are indexing along the K dimension (the last dim of obj_embeddings)
    # include_self=False ensures the initial zeros don't count towards the mean
    obj_embeddings.scatter_reduce_(
        dim=2, 
        index=index, 
        src=flat_feat, 
        reduce="sum", 
        include_self=False
    )
    assert obj_embeddings.shape == (B, C, K)
    
    # Permute to [B, K, C] for Transformer input
    obj_embeddings = obj_embeddings.permute(0, 2, 1)
    assert obj_embeddings.shape == (B, K, C)
    return obj_embeddings
    

class FourierFeatures2d(nn.Module):
    def __init__(self, num_freqs=10):
        super().__init__()
        self.num_freqs = num_freqs
        freqs = 2.0 ** torch.arange(num_freqs)
        self.register_buffer('freqs', freqs * torch.pi)

    def forward(self, x: torch.Tensor):
        #print(x.shape)
        scaled = x.unsqueeze(-1) * self.freqs
        #print(scaled.shape)
        scaled = scaled.view(*x.shape[:-1], -1) # (B, K, N, 2*num_freqs)
        #print(scaled.shape)
        sin_feats = torch.sin(scaled)
        cos_feats = torch.cos(scaled)
        return torch.cat([x, sin_feats, cos_feats], dim=-1)


class PolylineBackbone(nn.Module):
    def __init__(self, N, C, num_freqs=10):
        super().__init__()
        self.N = N
        self.C = C
        
        self.fourier = FourierFeatures2d(num_freqs=num_freqs)
        # 2 for untransformed input then 2 * num_freqs for sin and cos
        ff_dim = 2 + (4 * num_freqs) 
        
        self.mlp = nn.Sequential(
            nn.Linear(N * ff_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, C)
        )

    def forward(self, batch, mask=None):
        """
        x: (B, K_max, N, 2)
        mask: (B, K_max) boolean tensor, True for valid polylines.
        Returns: (B, K_max, C)
        """
        x = batch["lines"]
        B, K, N, _ = x.shape
        assert N == self.N
        
        x_ff = self.fourier(x) # (B, K, N, 2 + (4 * num_freqs) )
        #print(x_ff.shape)
        x_flat = x_ff.view(B, K, -1)
        #print(x_flat.shape)

        out = self.mlp(x_flat) # Shape: (B, K C)
        
        # --- NEW: Zero out the padded polylines ---
        if mask is not None:
            # mask shape is (B, K). We unsqueeze to (B, K, 1) to broadcast over C
            # ~mask inverts it (True becomes False). masked_fill_ puts 0s wherever ~mask is True.
            out = out.masked_fill(~mask.unsqueeze(-1), 0.0)
            
        return out

class ImageFeatureBackbone(nn.Module):
    def __init__(self, out_channels=128, max_objects: int = 1500):
        super().__init__()
        self.max_objects = max_objects
        # Load standard ResNet18
        # weights='IMAGENET1K_V1' gives good initial features
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        #resnet = models.resnet18()
        
        # We only take the early layers to keep high spatial resolution (Stride 4)
        # This preserves small objects better than the full 32x downsampling
        self.initial = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool
        )
        self.layer1 = resnet.layer1  # Stride remains 4
        # self.layer2 = resnet.layer2 # Optional: Stride 8, more semantic, less spatial precision
        
        # Projection to desired embedding dimension
        self.proj = nn.Conv2d(64, out_channels, kernel_size=1)
        self.out_channels = out_channels

    def forward(self, batch: dict[str, torch.Tensor])->torch.Tensor:
        # x: [B, C, H, W]
        x = batch["img"]
        seg_mask = batch["seg"]
        assert x.ndim ==4
        assert x.shape[1] ==3
        shape_start = x.shape
        
        x = self.initial(x)
        x = self.layer1(x)
        x = self.proj(x) # [B, out_channels, H/4, W/4]
        assert x.shape[1] == self.out_channels
        
        # UPSAMPLE features to match Mask Resolution (H, W)
        # This is memory intensive but necessary if objects are tiny and we can't scale mask down.
        # Bilinear is smoother for features.
        up = F.interpolate(x, scale_factor=4.0, mode='bilinear', align_corners=False)
        # if torch.isnan(up).any():
        #     raise RuntimeError("nan detected")
        assert up.shape[2:] == shape_start[2:]
        assert up.ndim == 4
        
        B, C, H, W = up.shape
        
        # 2. scatter mean
        # [B, C, H, W], [B, H, W] -> [B, K, C]
        obj_embeddings = scatter_2d(up, seg_mask, self.max_objects, check_bounds=True)
        
        return obj_embeddings


class RelationTransformer(nn.Module):
    def __init__(self, input_dim, num_layers=2, nhead=4):
        super().__init__()
        # Standard PyTorch implementation
        # batch_first=True makes input [Batch, Seq_Len, Dim]
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim, 
            nhead=nhead, 
            dim_feedforward=input_dim*4, 
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.input_dim = input_dim

    def forward(self, x, key_padding_mask=None):
        """
        x: [B, K, Dim]
        key_padding_mask: [B, K] (Bool: True means Ignore/Pad)
        """
        assert x.ndim == 3
        assert self.input_dim == x.shape[2]
        return self.transformer(x, src_key_padding_mask=key_padding_mask)


class RelationSetTransformer(nn.Module):
    def __init__(self, input_dim, num_layers=2, nhead=4, dropout=0.1):
        super().__init__()
        
        # We use the standard Encoder Layer.
        # Crucial: standard attention is permutation equivariant.
        # Only adding Positional Encodings (PE) breaks this.
        # PyTorch layers DO NOT add PE by default.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim, 
            nhead=nhead, 
            dim_feedforward=input_dim * 4, 
            batch_first=True,
            norm_first=True, # Pre-LN is generally more stable
            dropout=dropout
        )
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.input_dim = input_dim

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None):
        """
        x: [B, K, Dim] - A set of K elements
        key_padding_mask: [B, K] (Bool: True means Ignore/Pad)
        """
        assert x.ndim == 3
        assert x.shape[2] == self.input_dim

        # 1. We do NOT add Positional Encodings here. 
        #    This treats the input as a "Set" rather than a "Sequence".
        
        # 2. We pass src_mask=None explicitly (default) to ensure every 
        #    element can see every other element (All-to-All attention).
        
        # Note: In PyTorch 2.0+, scaled_dot_product_attention handles is_causal checks
        # automatically when mask is None.
        
        out = self.transformer(
            x, 
            src_key_padding_mask=key_padding_mask,
            mask=None,       # Explicitly no causal mask
            is_causal=False  # Explicit hint (PyTorch 2.0+)
        )
        
        return out


class ConstrainedOutputHead(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        # Simple projection for Query and Key
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)

    def forward(self, nodes: torch.Tensor, constraint_mask: torch.Tensor):
        """
        nodes: [B, K, d]
        constraint_mask: [B, K, K] (1=Valid, 0=Invalid)
        """
        # 1. Compute Alignment (Unnormalized Logits)
        # Q * K^T
        assert nodes.ndim ==3
        assert constraint_mask.ndim ==3
        assert nodes.shape[1] == constraint_mask.shape[1]
        

        
        K = nodes.shape[1]
        B = nodes.shape[0]
        d = nodes.shape[2]
        
        q = self.query(nodes)
        k = self.key(nodes)
        assert q.shape == (B, K, d)
        assert k.shape == (B, K, d)
        scores = torch.bmm(q, k.transpose(1, 2)) # [B, K, K]
        assert scores.shape == (B, K, K)
        
        # Scale factor
        scores = scores / (d ** 0.5)

        # 2. Apply Constraints "By Construction"
        # We need to mask where constraint_mask is 0 (False)
        bool_mask = constraint_mask.bool()
        #[B, K, 1]
        row_is_dead = (~bool_mask).all(dim=-1, keepdim=True)
        
        # Fill invalid positions with -inf
        # clone() is just safety for inplace ops, usually not strictly needed if masked_fill is out-of-place
        scores = scores.masked_fill(~bool_mask, float('-inf'))
        
        scores = scores.masked_fill(row_is_dead, 0.0)


        # 3. Softmax
        # Output sums to 1 over the last dimension
        probs = F.softmax(scores, dim=-1)
        assert probs.shape == (B, K, K)
        
        # 4. Cleanup NaNs
        # Rows that were fully -inf (no valid relations) become NaN after softmax.
        # We set them to 0.
        probs = probs.masked_fill(row_is_dead, 0.0)
        return probs


class SymmetricOutputHead(nn.Module):
    
    def __init__(self, input_dim):
        super().__init__()
        # Simple projection for Query and Key
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
    
    def forward(self, nodes: torch.Tensor, constraint_mask: torch.Tensor):
        """
        nodes: [B, K, d]
        constraint_mask: [B, K, K] (1=Valid, 0=Invalid)
        """
        # 1. Compute Alignment (Unnormalized Logits)
        assert nodes.ndim == 3
        assert constraint_mask.ndim == 3
        
        K = nodes.shape[1]
        B = nodes.shape[0]
        d = nodes.shape[2]
        
        q = self.query(nodes)
        k = self.key(nodes)
        
        # [B, K, K]
        scores = torch.bmm(q, k.transpose(1, 2)) 
        
        # Scale factor (acting as temperature)
        scores = scores / (d ** 0.5)
        

        # 2. Apply Constraints
        bool_mask = constraint_mask != 0
        
        # Sinkhorn requires these to remain -inf to ignore them correctly.
        scores = scores.masked_fill(~bool_mask, float('-inf'))

        scores_symm = 0.5*(scores + scores.transpose(1, 2))
        
        #scores_std = standardize_logits(scores_symm)

        # scores are treated as log-potentials. 
        # -inf entries will result in 0 mass in the output.
        # num_iters=20 is usually sufficient. 
        #probs = log_sinkhorn_iterations2(scores_symm, num_iters=20, epsilon=1.0)

        
        return scores_symm


class SceneGraphModelBase(nn.Module):
    def __init__(self, max_objects, d_dim=128):
        super().__init__()
        self.max_objects = max_objects
        self.d_dim = d_dim
        
        self.backbone = self._build_backbone()
        self.transformer = self._build_transformer()
        #self.head = ConstrainedOutputHead(d_dim)
        self.head = SymmetricOutputHead(d_dim)
    
    def _build_backbone(self) -> nn.Module:
        raise NotImplementedError()
    
    def _build_transformer(self) -> nn.Module:
        raise NotImplementedError()
        
    
    def _validate_object_pixels(self, seg_mask, n_objects_int):
        """
        Checks that every object index k < n_objects has at least 1 pixel in seg_mask.
        Raises RuntimeError if an expected object is missing from the mask.
        """
        B, H, W = seg_mask.shape
        K = self.max_objects
        
        # 1. Count pixels per object ID
        # Flatten: [B, H*W]
        flat_mask = seg_mask.view(B, -1)
        
        # We need to count occurrences of IDs 0..K-1
        # Create a container for counts: [B, K]
        # We create a one-hot-like encoding or just simple scatter add on ones
        ones = torch.ones_like(flat_mask, dtype=torch.float32)
        counts = torch.zeros(B, K, device=seg_mask.device)
        
        # We scatter_add the '1s' into the buckets defined by flat_mask
        # Note: We must ensure seg_mask doesn't have values >= K (padding/background handling)
        # Assuming clean data, but good to clamp or mask for this check
        safe_mask = flat_mask.clone()
        safe_mask[safe_mask >= K] = 0 # Redirect garbage to 0 just for counting safety
        
        # [B, K]
        counts.scatter_add_(1, safe_mask, ones)
        
        # 2. Identify violations
        # A violation is: index < n_objects AND count == 0
        
        # Create a mask of expected objects: [B, K]
        # True if index < n_objects
        range_vec = torch.arange(K, device=seg_mask.device).unsqueeze(0)
        expected_mask = range_vec < n_objects_int.unsqueeze(1)
        
        # Check counts
        # We specifically look at 'expected' slots.
        # If count is 0 there, it's a ghost object.
        zero_pixels = (counts == 0)
        violations = expected_mask & zero_pixels
        
        if violations.any():
            # Find the first culprit for the error message
            b_idx, k_idx = torch.where(violations)[0][0], torch.where(violations)[1][0]
            
            raise RuntimeError(
                f"NaN Hazard Detected! \n"
                f"Batch item {b_idx} expects {n_objects_int[b_idx]} objects.\n"
                f"However, Object ID {k_idx} has 0 pixels in the segmentation mask.\n"
                f"This causes division by zero in pooling and infinite gradients."
            )

    


    def forward(self, batch: dict[str, torch.Tensor]):
        """
        batch contains
        img: [B, 3, H, W]
        seg_mask: [B, H, W]
        constraint_matrix: [B, K, K]
        n_objects_int: [B] (Number of real objects)
        """
        
        B = batch["n_objects"].shape[0]
        constraint_matrix = batch["R"]
        K = constraint_matrix.shape[-1]
        assert constraint_matrix.ndim ==3
        
        # 1. Extract Dense Features
        obj_embeddings = self.backbone(batch)
        
        assert obj_embeddings.shape == (B, self.max_objects, self.d_dim)
        
        # probs = torch.bmm(obj_embeddings, obj_embeddings.transpose(1, 2))
        # assert probs.shape == (B, K, K)
        # 3. Create Padding Mask for Transformer
        # Transformer needs to know which tokens are padding. 
        # Mask is True for  padding indices.
        # Shape: [B, K]
        range_vec = torch.arange(K, device=obj_embeddings.device).unsqueeze(0)
        key_padding_mask = range_vec >=  batch["n_objects"].unsqueeze(1)
        assert key_padding_mask.shape == (B, K)
        
        # 4. Contextualize (Transformer)
        # [B, K, C]
        #print(obj_embeddings.shape)
        #print(key_padding_mask.shape)
        context_embeddings = self.transformer(obj_embeddings, key_padding_mask=key_padding_mask)
        
        #return context_embeddings
        # 5. Predict Relations (Constrained Head)
        # [B, K, K] (Softmaxed probabilities)
        probs : torch.Tensor = self.head(context_embeddings, constraint_matrix)
        return probs
        # probs_sm = 0.5 * (probs + probs.transpose(1, 2))
        # probs_sm = self.head(context_embeddings, constraint_matrix)
        # return probs_sm
        #print(n_objects_int)
        #print(seg_mask.max())
        #return probs_sm
        

class SceneGraphImageModel(SceneGraphModelBase):
    def _build_backbone(self):
        return ImageFeatureBackbone(out_channels=self.d_dim, max_objects=self.max_objects)
    def _build_transformer(self) -> nn.Module:
        return RelationSetTransformer(self.d_dim, num_layers=4, dropout=0.1)
    

class SceneGraphVectorModel(SceneGraphModelBase):
    def _build_backbone(self):
        return PolylineBackbone(100, self.d_dim, num_freqs=10)
    def _build_transformer(self) -> nn.Module:
        return RelationSetTransformer(self.d_dim, num_layers=4, dropout=0.1)