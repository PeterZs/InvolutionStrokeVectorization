import torch
import random


def scatter_with_repeats(src, m, total_placements=None):
    """
    src: tensor of shape (n,)
    m: destination size (m > n)
    total_placements: number of filled slots (>= n, <= m). 
                      If None, random between n and m.
    """
    n = src.numel()
    assert m > n

    if total_placements is None:
        total_placements = torch.randint(n, m + 1, (1,)).item()

    # --- choose where values will be placed in destination
    dest_indices = torch.randperm(m)[:total_placements]

    # --- guarantee each src value appears once
    mandatory = src

    # --- remaining placements are random repeats
    remaining = total_placements - n
    if remaining > 0:
        repeat_idx = torch.randint(0, n, (remaining,))
        repeats = src[repeat_idx]
        values = torch.cat([mandatory, repeats])
    else:
        values = mandatory

    # optional: shuffle placement order
    perm = torch.randperm(total_placements)
    dest_indices = dest_indices[perm]
    values = values[perm]

    # --- scatter into destination
    dest = torch.zeros(m, dtype=src.dtype, device=src.device)
    dest[dest_indices] = values

    return dest

def get_batched_padded_perm_matrices(sizes: torch.Tensor, perm_func, pad_size: int) -> torch.Tensor:
    """
    Generates a batch of random permutation matrices based on the Lehmer code generation,
    padded with zeros to a fixed size.

    Args:
        sizes (torch.Tensor): 1D Tensor containing the size n for each matrix in the batch.
        pad_size (int): The dimension to pad the matrices to (height and width).

    Returns:
        torch.Tensor: A tensor of shape (Batch, pad_size, pad_size).
    """
    batch_size = sizes.shape[0]
    device = sizes.device
    
    # Initialize the batch container with zeros
    # Shape: [Batch, Pad, Pad]
    batch_matrices = torch.zeros((batch_size, pad_size, pad_size), device=device)
    
    # We iterate through the batch because math.factorial and list operations 
    # are not easily vectorized in PyTorch for variable n.
    for b in range(batch_size):
        n = int(sizes[b].item())
        
        if n > pad_size:
            raise ValueError(f"Size at index {b} ({n}) exceeds pad_size ({pad_size})")
        
        if n > 0:
            # Get the random permutation sequence (e.g., [2, 0, 1])
            perm_indices = perm_func(n)
            
            # Convert to tensor for indexing
            perm_indices_tensor = torch.tensor(perm_indices, device=device, dtype=torch.long)
            
            # Create row indices [0, 1, ..., n-1]
            row_indices = torch.arange(n, device=device)
            
            # Set the corresponding coordinates to 1.
            # matrix[row, col] = 1
            batch_matrices[b, row_indices, perm_indices_tensor] = 1.0
            
    return batch_matrices




def permute_indices(indices: torch.Tensor, perm_matrices: torch.Tensor) -> torch.Tensor:
    """
    Permutes the values within an indices tensor to be consistent with 
    matrix multiplication y = Px.
    
    If perm_matrices[b, i, j] == 1, it means the mass from bin j moves to bin i.
    Therefore, all occurrences of value j in 'indices' are replaced by value i.

    Args:
        indices (torch.Tensor): [Batch, H, W] or [Batch, N] LongTensor.
        perm_matrices (torch.Tensor): [Batch, K, K] FloatTensor.

    Returns:
        torch.Tensor: Permuted indices of the same shape as input.
    """
    if indices.ndim not in [2, 3]:
        raise ValueError(f"indices must be 2D or 3D, got {indices.ndim}")
    
    B = indices.shape[0]
    K = perm_matrices.shape[1]
    device = indices.device
    
    # 1. Prepare the reference vector [0, 1, ..., K-1]
    # Shape: [B, K, 1]
    ref_seq = torch.arange(K, device=device, dtype=torch.float32)
    ref_seq = ref_seq.view(1, K, 1).expand(B, -1, -1)
    
    # 2. Extract the Mapping
    # We want mapping[j] = i where P[i, j] = 1.
    # To get this using matrix mult, we transpose P.
    # P_transpose[j, i] = 1. 
    # Row j of P_transpose has a 1 at column i.
    # Row j dot ref_seq -> returns i.
    # So mapping[j] = i.
    mapping_float = torch.bmm(perm_matrices.transpose(1, 2), ref_seq)
    
    # Shape: [B, K]
    mapping_table = mapping_float.squeeze(2).round().long()
    
    # 3. Apply Mapping via Gather
    # Flatten indices to [B, -1] for gather
    original_shape = indices.shape
    indices_flat = indices.view(B, -1)
    
    # Safety check (optional, can be removed for speed)
    if indices_flat.max() >= K:
        raise ValueError(f"Index {indices_flat.max()} out of bounds for matrix size {K}")
        
    # output[b, k] = mapping_table[b, indices_flat[b, k]]
    permuted_flat = torch.gather(input=mapping_table, dim=1, index=indices_flat)
    
    return permuted_flat.view(original_shape)

def random_involution(n: int) -> list[int]:
    """
    Generates a random involution of size n uniformly.
    An involution is a permutation where perm[perm[i]] == i.
    
    Args:
        n (int): The size of the permutation.
        
    Returns:
        list[int]: A list containing elements 0 to n-1.
    """
    if n == 0:
        return []
    if n == 1:
        return [0]

    # 1. Precompute Telephone Numbers (T)
    # T[i] stores the number of involutions of size i.
    # Formula: T(n) = T(n-1) + (n-1)*T(n-2)
    # We use integer arithmetic (Python handles large ints automatically).
    t_nums = [1] * (n + 1)
    for i in range(2, n + 1):
        t_nums[i] = t_nums[i-1] + (i - 1) * t_nums[i-2]

    # 2. Construct the involution
    perm = list(range(n))
    
    # We maintain a list of 'available' indices that haven't been fixed or swapped yet.
    available = list(range(n))
    
    while available:
        m = len(available)
        
        # If only 1 element remains, it must be a fixed point
        if m == 1:
            available.pop()
            break
            
        # Consider the last available element 'u'
        u = available[-1]
        
        # Calculate probability that 'u' is a fixed point.
        # Weight of being fixed: T(m-1)
        # Weight of being swapped: (m-1) * T(m-2)
        # Total Weight: T(m)
        # P(fixed) = T(m-1) / T(m)
        prob_fixed = t_nums[m-1] / t_nums[m]
        
        if random.random() < prob_fixed:
            # Case A: u is a fixed point (u -> u)
            # It's already set to itself in `perm` init, just remove from pool.
            available.pop()
        else:
            # Case B: u swaps with some other available element 'v'
            # We pick 'v' uniformly from the remaining (m-1) elements.
            # The indices in 'available' are 0 to m-2 (excluding u at m-1).
            v_index = random.randrange(m - 1)
            v = available[v_index]
            
            # Apply the swap
            perm[u] = v
            perm[v] = u
            
            # Remove u (last element)
            available.pop()
            
            # Remove v. To do this efficiently O(1), we move the 
            # last element of the list into v's spot and pop again.
            # Note: available[-1] is now the element that was at m-2 before u was popped.
            if v_index < len(available):
                available[v_index] = available[-1]
            available.pop()
            
    return perm

if __name__ == "__main__":
    # Batch size 2
    # Matrix size 5 (Indices 0-4 are valid)
    
    # Create dummy Permutation Matrices [2, 5, 5]
    # Batch 0 Mapping: [0->4, 1->3, 2->2, 3->1, 4->0] (Reverse)
    # Batch 1 Mapping: [0->1, 1->2, 2->0, 3->3, 4->4] (Shift first 3)

    perm_mat_0 = torch.eye(5).rot90(1)
    print(perm_mat_0)
    
    perm_mat_1 = torch.zeros((5, 5))
    perm_mat_1[0, 1] = 1; perm_mat_1[1, 2] = 1; perm_mat_1[2, 0] = 1; 
    perm_mat_1[3, 3] = 1; perm_mat_1[4, 4] = 1
    print(perm_mat_1)
    
    perm_matrices = torch.stack([perm_mat_0, perm_mat_1])
    
    # Input Indices [2, 2, 3] (Batch=2, H=2, W=3)
    # Batch 0 inputs: values 0, 1, 4
    # Batch 1 inputs: values 0, 1, 2
    input_indices = torch.tensor([
        [[0, 1, 4], 
         [4, 1, 0]], 
        
        [[0, 1, 2], 
         [2, 1, 0]]
    ], dtype=torch.long)
    
    output = permute_indices(input_indices, perm_matrices)
    
    print("--- Batch 0 ---")
    print("Mapping: 0->4, 1->3, 4->0")
    print("Input:\n", input_indices[0])
    print("Output:\n", output[0])
    # Expected Output Batch 0:
    # [[4, 3, 0],
    #  [0, 3, 4]]
    
    print("\n--- Batch 1 ---")
    print("Mapping: 0->1, 1->2, 2->0")
    print("Input:\n", input_indices[1])
    print("Output:\n", output[1])
    # Expected Output Batch 1:
    # [[1, 2, 0],
    #  [0, 2, 1]]