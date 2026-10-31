# Portions derived from PX4, BSD 3 Clause. See THIRD_PARTY_LICENSES.


import numpy as np


class PseudoInverse:
    def __init__(self):
        pass

    def full_rank_cholesky(self, A):
        N = A.shape[0]
        tol = N * np.finfo(A.dtype).eps * np.max(np.diag(A))
        L = np.zeros_like(A)
        rank = 0

        for k in range(N):
            if rank == 0:
                L[k:, rank] = A[k:, k]
            else:
                LL = L[k:, :rank] @ L[k, :rank].T
                L[k:, rank] = A[k:, k] - LL

            if L[k, rank] > tol:
                L[k, rank] = np.sqrt(L[k, rank])
                if k < N - 1:
                    L[k + 1 :, rank] /= L[k, rank]
                rank += 1

        return L[:, :rank], rank

    def geninv(self, G):
        M, N = G.shape

        if M <= N:
            A = G @ G.T
            L, rank = self.full_rank_cholesky(A)
            A = L.T @ L

            try:
                X = np.linalg.inv(A)
            except np.linalg.LinAlgError:
                return np.zeros((N, M))

            A = X @ X @ L.T
            res = G.T @ (L @ A)
        else:
            A = G.T @ G
            L, rank = self.full_rank_cholesky(A)
            A = L.T @ L

            try:
                X = np.linalg.inv(A)
            except np.linalg.LinAlgError:
                return np.zeros((N, M))

            A = X @ X @ L.T
            res = (L @ A) @ G.T

        return res

    def normalize_control_allocation_matrix(self, M):
        scale = np.zeros(M.shape[1])

        if np.any(np.abs(M[:, 0]) > 1e-3):
            scale[0] = np.sqrt(
                np.sum(M[:, 0] ** 2) / (np.count_nonzero(np.abs(M[:, 0]) > 1e-3) / 2.0)
            )

        if np.any(np.abs(M[:, 1]) > 1e-3):
            scale[1] = np.sqrt(
                np.sum(M[:, 1] ** 2) / (np.count_nonzero(np.abs(M[:, 1]) > 1e-3) / 2.0)
            )

        max_scale = max(scale[0], scale[1]) if max(scale[0], scale[1]) > 0 else 1.0
        M[:, 0] /= max_scale
        M[:, 1] /= max_scale

        yaw_max = np.max(np.abs(M[:, 2]))
        if yaw_max > 0:
            M[:, 2] /= yaw_max

        thrust_max = np.max(np.abs(M[:, 5]))
        if thrust_max > 0:
            M[:, 5] /= thrust_max

        return M

    def update_control_allocation_matrix_scale(self, M):
        num_non_zero_roll = np.count_nonzero(np.abs(M[:, 0]) > 1e-3)
        num_non_zero_pitch = np.count_nonzero(np.abs(M[:, 1]) > 1e-3)

        roll_norm_scale = 1.0
        pitch_norm_scale = 1.0

        if num_non_zero_roll > 0:
            roll_norm_scale = np.sqrt(np.sum(M[:, 0] ** 2) / (num_non_zero_roll / 2.0))

        if num_non_zero_pitch > 0:
            pitch_norm_scale = np.sqrt(
                np.sum(M[:, 1] ** 2) / (num_non_zero_pitch / 2.0)
            )

        control_allocation_scale = max(roll_norm_scale, pitch_norm_scale)

        M[:, 0] /= control_allocation_scale
        M[:, 1] /= control_allocation_scale

        yaw_max = np.max(M[:, 2])
        if yaw_max > 0:
            M[:, 2] /= yaw_max

        return M

    def update_pseudo_inverse_full(self, effectiveness_matrix):
        mix = self.geninv(effectiveness_matrix)
        print("Computed Pseudo-Inverse (_mix):")
        print(mix)

        mix_scaled = self.update_control_allocation_matrix_scale(mix.copy())
        print("\nAfter scale (_mix):")
        print(mix_scaled)

        mix_normalized = self.normalize_control_allocation_matrix(mix_scaled.copy())
        print("\nAfter normalized (_mix):")
        print(mix_normalized)

        return mix_normalized
