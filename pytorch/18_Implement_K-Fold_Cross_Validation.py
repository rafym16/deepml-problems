import torch
from typing import List, Tuple


def k_fold_cross_validation_torch(n_samples: int, k: int = 5, shuffle: bool = True) -> List[
    Tuple[List[int], List[int]]]:
    """
    Return train/test index splits for k-fold cross-validation menggunakan PyTorch.

    Args:
        n_samples: Total jumlah sampel dalam dataset.
        k: Jumlah fold.
        shuffle: Apakah indeks diacak sebelum dibagi.

    Returns:
        List berisi tuple (train_indices, test_indices) dalam bentuk list of ints.
    """
    # 1. Inisialisasi indeks
    indices = torch.arange(n_samples)

    # 2. Pengacakan (Shuffle)
    if shuffle:
        # torch.randperm menghasilkan permutasi acak dari 0 sampai n_samples-1
        indices = indices[torch.randperm(n_samples)]

    fold_size = n_samples // k
    splits = []

    for i in range(k):
        # 3. Tentukan batas indeks untuk test set
        start = i * fold_size
        # Pastikan fold terakhir mengambil sisa data jika n_samples tidak habis dibagi k
        end = n_samples if i == k - 1 else (i + 1) * fold_size

        test_indices = indices[start:end]

        # 4. Gabungkan sisa indeks untuk train set
        # Mengambil bagian sebelum 'start' dan sesudah 'end'
        train_indices = torch.cat((indices[:start], indices[end:]))

        # 5. Konversi kembali ke list of ints sesuai permintaan template
        splits.append((train_indices.tolist(), test_indices.tolist()))

    return splits