import numpy as np
import torch
import torch.nn.functional as F


class ICSMetric:
    """
    Inter-Class Similarity (ICS) metrics using k-nearest neighbors.

    Supports:
    - Single video with multiple scenes: [N_scenes, N_scene_frames, d]
    - Multiple videos with multiple scenes: [N_videos, N_scenes, N_scene_frames, d]
    """

    def __init__(
        self,
        k: int = 10,
        n_segments: int = 30,
        similarity_threshold: float = 0.85,
        metric: str = 'cosine',
        normalize: bool = True,
        batch_size: int = 1000,
        n_chunks: int = 5,
        device_id: str = 'cuda:0'
    ):
        self.k = k
        self.n_segments = n_segments
        self.similarity_threshold = similarity_threshold
        self.metric = metric.lower()
        if self.metric == 'euclidean':
            self.metric = 'l2'
        self.normalize = normalize
        self.batch_size = batch_size
        self.n_chunks = n_chunks
        self.device = torch.device(
            device_id if torch.cuda.is_available() and 'cuda' in device_id else 'cpu'
        )

    # ---------------- Segment computation ----------------
    def _compute_segment_embeddings_single(self, video_features: np.ndarray) -> np.ndarray:
        """Split single video (or scene) into segments and average features."""
        N, d = video_features.shape
        if self.n_segments >= N:
            return video_features

        bounds = np.linspace(0, N, self.n_segments + 1, dtype=int)
        bounds = np.unique(bounds)
        sums = np.add.reduceat(video_features, bounds[:-1], axis=0)
        sizes = np.diff(bounds).reshape(-1, 1)
        return sums / sizes

    def _compute_segment_embeddings(self, embeddings) -> np.ndarray:
        """
        Supports:

        1 video:
            list[np.ndarray] where each = [N_frames, d]

        Multiple videos:
            list[list[np.ndarray]]

        Returns:
            [total_segments, d]
        """

        all_segments = []

        # Case 1: list of scenes (one video)
        if isinstance(embeddings, list) and isinstance(embeddings[0], np.ndarray):

            for scene in embeddings:
                segments = self._compute_segment_embeddings_single(scene)
                all_segments.append(segments)

        # Case 2: list of videos -> list of scenes
        elif isinstance(embeddings, list) and isinstance(embeddings[0], list):

            for video in embeddings:
                for scene in video:
                    segments = self._compute_segment_embeddings_single(scene)
                    all_segments.append(segments)

        # Case 3: numpy array (already padded)
        elif isinstance(embeddings, np.ndarray):

            if embeddings.ndim == 3:
                # [N_scenes, N_frames, d]
                for scene in embeddings:
                    segments = self._compute_segment_embeddings_single(scene)
                    all_segments.append(segments)

            elif embeddings.ndim == 4:
                # [N_videos, N_scenes, N_frames, d]
                for video in embeddings:
                    for scene in video:
                        segments = self._compute_segment_embeddings_single(scene)
                        all_segments.append(segments)

            else:
                raise ValueError(f"Unsupported shape: {embeddings.shape}")

        else:
            raise TypeError("Unsupported embeddings format")

        if not all_segments:
            raise ValueError("No segments extracted")

        return np.concatenate(all_segments, axis=0)

    # ---------------- Merge similar segments ----------------
    def _merge_similar_segments(self, segment_embeddings: np.ndarray) -> np.ndarray:
        if len(segment_embeddings) <= 1:
            return segment_embeddings

        embeddings = torch.from_numpy(segment_embeddings).float()
        threshold = self.similarity_threshold if self.metric == 'cosine' else self.similarity_threshold * 10

        def should_merge(emb1: torch.Tensor, emb2: torch.Tensor) -> bool:
            if self.metric == 'cosine':
                sim = F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()
                return sim >= threshold
            else:
                dist = torch.norm(emb1 - emb2).item()
                return dist <= threshold

        while True:
            merged = []
            current_group = [embeddings[0]]
            has_merged = False

            for i in range(1, len(embeddings)):
                if should_merge(current_group[-1], embeddings[i]):
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

    # ---------------- Normalization ----------------
    def _normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-8, norms)
        return embeddings / norms

    # ---------------- k-NN distances ----------------
    @torch.no_grad()
    def _compute_distances(self, embeddings: np.ndarray, k: int) -> np.ndarray:
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.ndim != 2:
            embeddings = embeddings.reshape(embeddings.shape[0], -1)

        embeddings_tensor = torch.from_numpy(embeddings).float()
        N, d = embeddings_tensor.shape

        if self.metric == 'cosine':
            embeddings_tensor = F.normalize(embeddings_tensor, p=2, dim=1, eps=1e-8)

        all_distances = []

        chunk_size = (N + self.n_chunks - 1) // self.n_chunks
        db_chunks = [(i, min(i + chunk_size, N)) for i in range(0, N, chunk_size)]

        for i in range(0, N, self.batch_size):
            batch_end = min(i + self.batch_size, N)
            query_batch = embeddings_tensor[i:batch_end].to(self.device)
            batch_distances = []

            for start, end in db_chunks:
                db_chunk = embeddings_tensor[start:end].to(self.device)
                if self.metric == 'cosine':
                    sims = torch.mm(query_batch, db_chunk.T)
                    distances = torch.clamp(1.0 - sims, min=0.0, max=2.0)
                else:
                    query_norm_sq = (query_batch ** 2).sum(dim=1, keepdim=True)
                    db_norm_sq = (db_chunk ** 2).sum(dim=1)
                    dot = torch.mm(query_batch, db_chunk.T)
                    sq_distances = query_norm_sq + db_norm_sq.unsqueeze(0) - 2 * dot
                    distances = torch.sqrt(torch.clamp(sq_distances, min=0.0))

                batch_distances.append(distances)

            distances_full = torch.cat(batch_distances, dim=1)
            knn_dist, _ = torch.topk(distances_full, k=k + 1, dim=1, largest=False, sorted=True)
            knn_dist = knn_dist[:, 1:]  # Remove self-distance
            all_distances.append(knn_dist.cpu())

        return torch.cat(all_distances, dim=0).numpy()

    # ---------------- Compute ICS ----------------
    def compute(self, embeddings: list[np.ndarray | list]) -> dict:
        # 1. Segment embeddings
        processed = self._compute_segment_embeddings(embeddings)
        # 2. Merge similar segments
        processed = self._merge_similar_segments(processed)
        # 3. Normalize
        if self.normalize:
            processed = self._normalize_embeddings(processed)

        print(processed.shape)
        total_segments = processed.shape[0]

        # 4. k-NN
        k = self.k if self.k is not None else total_segments - 1
        print(k)
        k = min(k, total_segments - 1)

        if k <= 0:
            return {
                'mean_knn_dist': 0.0,
                'min_observed': 0.0,
                'max_observed': 0.0,
                'total_segments': total_segments
            }

        knn_distances = self._compute_distances(processed, k)

        return {
            'mean_knn_dist': float(knn_distances.mean(axis=1).mean()),
            'min_observed': float(knn_distances.min()),
            'max_observed': float(knn_distances.max()),
            'total_segments': total_segments
        }


if __name__ == "__main__":
    dataset_features = []
    for file in ("5aa1781c7a2db27aedc6ce20a520afe4b01d06f082b4db83deb20ff745d0fbe6_features.npy",
                 "d589826a4eef5b5f14a1ba1a27db450c1694c62c78e934ffab5b40db948d998d_features.npy",
                 "68eaff17a049a45efc5e889fab28ac478d340f0c519bceabea15a974188db4be_features.npy",
                 "116de85ffc503ffe2de160c9543fe6665facd6d8801fa4181627ce1418ff243f_features.npy",
                 "534caa39ec7549a34161eb21242043c1da57e445588c98bcb70ac275fcf667ea_features.npy",
                 "0595fb9e501554e2323be64b30b9b3d9c2c2bd2a2604d7922f43e6e13d226eb7_features.npy",
                 "901cee0fe2f5cf7f30431f157acd5176001af952d8c189da26768afb085e1f63_features.npy",
                 "26129689747d638df034ff2a15d1b6fbe1d3600467f01b80ff8d847577da632d_features.npy",
                 "d589826a4eef5b5f14a1ba1a27db450c1694c62c78e934ffab5b40db948d998d_features.npy",
                 "c00edeba6d8b6fc870f09df3d3aecbd3ca90d0ee2252dbb53322bccc7ad206c5_features.npy"
                 ):
        feature_file = f"../../.analysis_cache_industrial_v2/{file}"
        video_features = np.load(feature_file)
        dataset_features.append(video_features)
    print(ICSMetric(normalize=True).compute(dataset_features))