def gridworld_policy_evaluation(policy, gamma, threshold):
    size = 5  # ukuran grid 5x5

    # Inisialisasi value function (semua state = 0)
    V = [[0.0 for _ in range(size)] for _ in range(size)]

    # Fungsi untuk cek apakah state adalah terminal (pojok)
    def is_terminal(i, j):
        return (i == 0 and j == 0) or \
            (i == 0 and j == size - 1) or \
            (i == size - 1 and j == 0) or \
            (i == size - 1 and j == size - 1)

    # Fungsi untuk menentukan next state berdasarkan action
    def get_next_state(i, j, action):
        if action == 'up':
            return (max(i - 1, 0), j)  # tidak bisa keluar grid
        elif action == 'down':
            return (min(i + 1, size - 1), j)
        elif action == 'left':
            return (i, max(j - 1, 0))
        elif action == 'right':
            return (i, min(j + 1, size - 1))

    # Iterasi sampai konvergen
    while True:
        delta = 0  # perubahan maksimum di setiap iterasi

        # copy value lama (biar update tidak saling ganggu)
        new_V = [row[:] for row in V]

        # Loop semua state
        for i in range(size):
            for j in range(size):

                # Kalau terminal → skip (value tetap 0)
                if is_terminal(i, j):
                    continue

                v = 0.0  # value baru untuk state (i, j)

                # Loop semua action sesuai policy
                for action, prob in policy[(i, j)].items():
                    # Cari next state
                    ni, nj = get_next_state(i, j, action)

                    reward = -1  # reward konstan

                    # Bellman expectation update
                    v += prob * (reward + gamma * V[ni][nj])

                # Simpan value baru
                new_V[i][j] = v

                # Hitung perubahan terbesar
                delta = max(delta, abs(v - V[i][j]))

        # Update V
        V = new_V

        # Kalau sudah konvergen → berhenti
        if delta < threshold:
            break

    return V