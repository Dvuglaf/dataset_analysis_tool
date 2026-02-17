import numpy as np
import torch
from sklearn.cluster import DBSCAN

import numpy as np
import torch
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors


class ClusterMetric:
    def __init__(
            self,
            eps: float = None,  # Если None, подбираем автоматически
            min_samples: int = 5,
            n_segments: int = 30,
            merge_threshold: float = 8.5,
            batch_size: int = 1000,
            n_chunks: int = 5,
            normalize: bool = False,
            device_id: str = 'cuda:0'
    ):
        self.eps = eps
        self.n_chunks = n_chunks
        self.batch_size = batch_size
        self.min_samples = min_samples
        self.n_segments = n_segments
        self.merge_threshold = merge_threshold
        self.normalize = normalize
        self.device = torch.device(device_id if torch.cuda.is_available() else 'cpu')

    def _ensure_list_of_frames(self, embeddings):
        """
        Приводит любой вход к списку [N_frames, 512].
        Обрабатывает:
        - (N_scenes, N_frames, 512) -> склеивает сцены в одно видео
        - (N_videos, N_scenes, N_frames, 512) -> каждое видео в список
        - Списки массивов разной длины
        """
        output = []

        # Если это одиночный массив (numpy или torch)
        if isinstance(embeddings, (np.ndarray, torch.Tensor)):
            if isinstance(embeddings, torch.Tensor):
                embeddings = embeddings.detach().cpu().numpy()

            # Case: (N_scenes, N_frames, 512)
            if embeddings.ndim == 3:
                # Склеиваем сцены в одну последовательность кадров
                video = embeddings.reshape(-1, embeddings.shape[-1])
                output.append(video)

            # Case: (N_videos, N_scenes, N_frames, 512)
            elif embeddings.ndim == 4:
                for v in range(embeddings.shape[0]):
                    video = embeddings[v].reshape(-1, embeddings.shape[-1])
                    output.append(video)

            elif embeddings.ndim == 2:
                output.append(embeddings)

        # Если это список (самый гибкий вариант для разных длин)
        elif isinstance(embeddings, list):
            for item in embeddings:
                # Рекурсивно вызываем для каждого элемента списка
                output.extend(self._ensure_list_of_frames(item))

        return output

    def _find_optimal_eps(self, embeddings: np.ndarray):
        """Автоматический подбор eps через метод K-ближайших соседей."""
        n_samples = embeddings.shape[0]
        k = min(self.min_samples, n_samples - 1)
        # k берем равным min_samples
        neigh = NearestNeighbors(n_neighbors=k)
        neigh.fit(embeddings)
        distances, _ = neigh.kneighbors(embeddings)

        # Берем расстояния до k-го соседа
        k_distances = np.sort(distances[:, -1])

        # Находим точку максимального изгиба (простой эвристический метод)
        # В идеале это 'elbow' на графике
        # Для простоты возьмем 80-й или 90-й перцентиль как безопасную базу
        optimal_eps = np.percentile(k_distances, 80)

        # Если eps получился слишком маленьким (0), берем среднее
        if optimal_eps < 1e-5:
            optimal_eps = k_distances.mean()

        return max(float(optimal_eps), 0.1)

    def _compute_segment_embeddings(self, embeddings_list) -> np.ndarray:
        all_segments = []
        for video_features in embeddings_list:
            # Сегментируем видео (N_frames, 512) -> (n_segments, 512)
            segments = self._compute_segment_embeddings_single(video_features)
            # Мержим похожие (n_segments, 512) -> (M, 512)
            segments = self._merge_similar_segments(segments)
            all_segments.append(segments)

        # КРИТИЧНО: Конкатенируем все сегменты всех видео в одну общую базу
        return np.concatenate(all_segments, axis=0)

    def _compute_segment_embeddings_single(self, video_features: np.ndarray) -> np.ndarray:
        """Split single video into segments and average features within each segment.

        Args:
            video_features: [N_frames, d] - features for one video

        Returns:
            [actual_segments, d] - segment embeddings (actual_segments <= n_segments)
        """
        if video_features.ndim != 2:
            raise ValueError(f"Expected 2D array [N_frames, d], got shape {video_features.shape}")

        N, d = video_features.shape

        if N == 0:
            return np.empty((0, d), dtype=video_features.dtype)

        if self.n_segments >= N:
            return video_features

        segment_bounds = np.linspace(0, N, self.n_segments + 1, dtype=int)
        segment_bounds = np.unique(segment_bounds)

        actual_segments = len(segment_bounds) - 1
        if actual_segments == 0:
            return video_features

        segment_sums = np.add.reduceat(video_features, segment_bounds[:-1], axis=0)
        segment_sizes = np.diff(segment_bounds).reshape(-1, 1)

        return segment_sums / segment_sizes

    def _merge_similar_segments(self, segment_embeddings: np.ndarray) -> np.ndarray:
        """Iteratively merge consecutive similar segments within a single video.

        Args:
            segment_embeddings: [N, d] - segment embeddings from one video

        Returns:
            [M, d] - merged embeddings where M <= N
        """
        if segment_embeddings.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {segment_embeddings.shape}")

        if len(segment_embeddings) <= 1:
            return segment_embeddings

        embeddings = torch.from_numpy(segment_embeddings).float()
        threshold = self.merge_threshold

        while True:
            merged = []
            current_group = [embeddings[0]]
            has_merged = False

            for i in range(1, len(embeddings)):
                distance = torch.norm(current_group[-1] - embeddings[i]).item()
                if distance <= threshold:
                    current_group.append(embeddings[i])
                    has_merged = True
                else:
                    merged.append(torch.stack(current_group).mean(dim=0))
                    current_group = [embeddings[i]]

            merged.append(torch.stack(current_group).mean(dim=0))
            embeddings = torch.stack(merged)

            if not has_merged or len(embeddings) <= 1:
                break

        return embeddings.numpy()

    def _normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """L2 normalization."""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-8, norms)
        return embeddings / norms

    @torch.no_grad()
    def _compute_distance_matrix(self, embeddings: np.ndarray) -> np.ndarray:
        """Compute full pairwise L2 distance matrix using GPU."""
        if embeddings.ndim == 3 and embeddings.shape[1] == 1:
            embeddings = embeddings.squeeze(1)  # (N,1,512) -> (N,512)

        if embeddings.ndim != 2:
            raise ValueError(f"Expected (N,D), got {embeddings.shape}")
        embeddings_tensor = torch.from_numpy(embeddings).float()
        N = embeddings_tensor.shape[0]

        embeddings_norm_sq = (embeddings_tensor ** 2).sum(dim=1)

        chunk_size = (N + self.n_chunks - 1) // self.n_chunks
        db_chunks = [(i, min(i + chunk_size, N)) for i in range(0, N, chunk_size)]

        all_distances = []

        for i in range(0, N, self.batch_size):
            batch_end = min(i + self.batch_size, N)
            query_batch = embeddings_tensor[i:batch_end].to(self.device)
            query_norm_sq = (query_batch ** 2).sum(dim=1, keepdim=True)

            batch_distances = []

            for chunk_start, chunk_end in db_chunks:
                db_chunk = embeddings_tensor[chunk_start:chunk_end].to(self.device)
                db_norm_sq = embeddings_norm_sq[chunk_start:chunk_end].to(self.device)

                dot_product = torch.mm(query_batch, db_chunk.T)
                sq_distances = query_norm_sq + db_norm_sq.unsqueeze(0) - 2.0 * dot_product
                distances_chunk = torch.sqrt(torch.clamp(sq_distances, min=0.0))

                batch_distances.append(distances_chunk)

                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()

            distances_full = torch.cat(batch_distances, dim=1)
            all_distances.append(distances_full.cpu())

            if self.device.type == 'cuda':
                torch.cuda.empty_cache()

        return torch.cat(all_distances, dim=0).numpy()

    def _compute_cluster_radius(self, embeddings: np.ndarray, labels: np.ndarray) -> dict:
        """Compute radius of each cluster (mean L2 distance from centroid to points).

        Args:
            embeddings: [N, d] - embeddings
            labels: [N] - cluster labels from DBSCAN

        Returns:
            dict with cluster_radii, mean_radius, num_clusters
        """
        if embeddings.shape[0] != labels.shape[0]:
            raise ValueError(f"Shape mismatch: embeddings {embeddings.shape[0]} vs labels {labels.shape[0]}")

        unique_labels = set(labels)
        unique_labels.discard(-1)

        cluster_radii = {}

        for label in unique_labels:
            mask = labels == label
            cluster_embeddings = embeddings[mask]

            if len(cluster_embeddings) < 2:
                cluster_radii[label] = 0.0
                continue

            centroid = cluster_embeddings.mean(axis=0)
            distances_to_centroid = np.linalg.norm(cluster_embeddings - centroid, axis=1)
            cluster_radii[label] = float(distances_to_centroid.mean())

        mean_radius = np.mean(list(cluster_radii.values())) if cluster_radii else 0.0

        return {
            'cluster_radii': cluster_radii,
            'mean_radius': mean_radius,
            'num_clusters': len(cluster_radii)
        }

    def _compute_inter_cluster_distance(self, embeddings: np.ndarray, labels: np.ndarray) -> dict:
        """Compute mean L2 distance between cluster centroids."""
        unique_labels = set(labels)
        unique_labels.discard(-1)

        if len(unique_labels) < 2:
            return {'mean_inter_cluster_distance': 0.0}

        centroids = {}
        for label in unique_labels:
            mask = labels == label
            centroids[label] = embeddings[mask].mean(axis=0)

        labels_list = list(unique_labels)
        distances = []

        for i in range(len(labels_list)):
            for j in range(i + 1, len(labels_list)):
                dist = np.linalg.norm(centroids[labels_list[i]] - centroids[labels_list[j]])
                distances.append(dist)

        return {'mean_inter_cluster_distance': np.mean(distances) if distances else 0.0}

    def compute(self, embeddings) -> dict:
        # 1. Приводим к единому формату
        frames_list = self._ensure_list_of_frames(embeddings)

        # 2. Получаем эмбеддинги сегментов (общий массив для всех видео)
        processed = self._compute_segment_embeddings(frames_list)

        if self.normalize:
            processed = self._normalize_embeddings(processed)

        # 3. Вычисляем матрицу расстояний
        distance_matrix = self._compute_distance_matrix(processed)

        # 4. АВТО-EPS: если не задан, считаем на лету
        current_eps = self.eps if self.eps is not None else self._find_optimal_eps(processed)

        # 5. Кластеризация
        dbscan = DBSCAN(eps=current_eps, min_samples=self.min_samples, metric='precomputed',
                        n_jobs=-1)
        labels = dbscan.fit_predict(distance_matrix)

        # Compute statistics
        unique_labels = set(labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        n_noise = (labels == -1).sum()
        noise_ratio = n_noise / len(labels) if len(labels) > 0 else 0.0

        # Compute metrics
        radius_metrics = self._compute_cluster_radius(processed, labels)
        inter_cluster_metrics = self._compute_inter_cluster_distance(processed, labels)

        mean_radius = radius_metrics['mean_radius']
        mean_inter_cluster_dist = inter_cluster_metrics['mean_inter_cluster_distance']

        if mean_inter_cluster_dist > 0:
            r_R_score = mean_radius / mean_inter_cluster_dist
        else:
            r_R_score = float('inf')

        return {
            'mean_cluster_radius': mean_radius,
            'mean_inter_cluster_distance': mean_inter_cluster_dist,
            'r_R_score': r_R_score,
            'num_clusters': n_clusters,
            'noise_ratio': noise_ratio,
            'total_segments': processed.shape[0],
            'eps': self.eps,
            'min_samples': self.min_samples
        }


class ClusterAMetric:
    """
    Cluster metrics using DBSCAN on video segment embeddings.

    Computes cluster radius, inter-cluster distance, and r/R score
    to measure how well-separated different actions are in embedding space.
    """

    def __init__(
        self,
        eps: float = 2.3,
        min_samples: int = 2,
        n_segments: int = 30,
        merge_threshold: float = 8.5,
        normalize: bool = False,
        batch_size: int = 1000,
        n_chunks: int = 5,
        device_id: str = 'cuda:0'
    ):
        """
        Args:
            eps: DBSCAN eps parameter (max distance between neighbors)
            min_samples: DBSCAN min_samples parameter
            n_segments: number of segments to split each video into
            merge_threshold: L2 distance threshold for merging similar segments
            normalize: whether to L2 normalize embeddings
            batch_size: batch size for GPU processing
            n_chunks: number of chunks to split database into
            device_id: device for computation
        """
        self.eps = eps
        self.min_samples = min_samples
        self.n_segments = n_segments
        self.merge_threshold = merge_threshold
        self.normalize = normalize
        self.batch_size = batch_size
        self.n_chunks = n_chunks
        self.device = torch.device(
            device_id if torch.cuda.is_available() and 'cuda' in device_id else 'cpu'
        )



    def _compute_segment_embeddings(self, embeddings) -> np.ndarray:
        """Process all videos: compute segments and merge within each video.

        Args:
            embeddings: list of [N_frames_i, d] arrays or [N_videos, N_frames, d] array

        Returns:
            [total_segments, d] - concatenated segment embeddings from all videos
        """
        all_segments = []
        for video_features in embeddings:
            video_features = np.asarray(video_features, dtype=np.float32)

            # Compute segments for this video
            segments = self._compute_segment_embeddings_single(video_features)

            # Merge similar segments within this video only
            segments = self._merge_similar_segments(segments)

            all_segments.append(segments)

        return np.concatenate(all_segments, axis=0)



    def compute(self, embeddings) -> dict:
        """
        Compute cluster metrics.

        Args:
            embeddings: list of [N_frames_i, d] arrays or [N_videos, N_frames, d] array
                        - video embeddings (per-frame features for each video)

        Returns:
            dict with cluster metrics:
                - mean_cluster_radius: mean radius of clusters
                - mean_inter_cluster_distance: mean distance between cluster centers
                - r_R_score: ratio of radius to inter-cluster distance
                - num_clusters: number of clusters found
                - noise_ratio: fraction of noise points
                - total_segments: number of segments after processing
                - eps: DBSCAN eps parameter used
                - min_samples: DBSCAN min_samples parameter used
        """
        # Preprocess embeddings
        if isinstance(embeddings, np.ndarray):
            processed = embeddings.astype(np.float32)
        else:
            processed = embeddings  # list handled in _compute_segment_embeddings

        processed = self._compute_segment_embeddings(processed)

        if self.normalize:
            processed = self._normalize_embeddings(processed)

        distance_matrix = self._compute_distance_matrix(processed)
        print("min dist:", distance_matrix.min())
        print("mean dist:", distance_matrix.mean())
        print("max dist:", distance_matrix.max())

        # Fit DBSCAN
        dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples, metric='precomputed', n_jobs=-1)
        labels = dbscan.fit_predict(distance_matrix)


