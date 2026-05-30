import torch
import torch.nn.functional as F

from src.tta.base import BaseTTA


class CentroidPseudoLabelTTA(BaseTTA):
    """
    Reassign predictions by nearest soft-probability-weighted class centroid.
    """

    def __init__(
        self,
        model,
        num_classes=2,
        distance="cosine",
        logit_scale=10.0,
        eps=1e-8,
    ):
        super().__init__(model)
        self.num_classes = num_classes
        self.distance = distance
        self.logit_scale = logit_scale
        self.eps = eps
        self.centroids = None

    def before_partition(self, dataloader, move_batch_to_device, transform_batch):
        self.model.eval()
        sums = None
        weights = None

        with torch.no_grad():
            for batch in dataloader:
                batch = transform_batch(move_batch_to_device(batch))
                outputs = self.model(**batch)
                embeddings = outputs["embeddings"]
                probabilities = outputs["logits"].softmax(dim=-1)

                if sums is None:
                    sums = embeddings.new_zeros(self.num_classes, embeddings.shape[-1])
                    weights = embeddings.new_zeros(self.num_classes)

                sums += probabilities.transpose(0, 1) @ embeddings
                weights += probabilities.sum(dim=0)

        self.centroids = sums / weights.clamp_min(self.eps).unsqueeze(-1)
        centroid_norms = self.centroids.norm(dim=-1)
        return {
            "centroid_min_weight": weights.min().item(),
            "centroid_max_weight": weights.max().item(),
            "centroid_mean_weight": weights.mean().item(),
            "centroid_mean_norm": centroid_norms.mean().item(),
        }

    def predict_batch(self, batch):
        if self.centroids is None:
            raise RuntimeError(
                "Centroids are not initialized. Call before_partition first."
            )

        with torch.no_grad():
            outputs = self.model(**batch)
            embeddings = outputs["embeddings"]
            if self.distance == "cosine":
                norm_embeddings = F.normalize(embeddings, dim=-1)
                norm_centroids = F.normalize(self.centroids, dim=-1)
                logits = self.logit_scale * (norm_embeddings @ norm_centroids.T)
            elif self.distance == "euclidean":
                logits = -self.logit_scale * torch.cdist(embeddings, self.centroids)
            else:
                raise ValueError(f"Unknown centroid distance: {self.distance}")

            outputs["base_logits"] = outputs["logits"]
            outputs["logits"] = logits
            probabilities = logits.softmax(dim=-1)
            top_probabilities = probabilities.topk(
                k=min(2, probabilities.shape[-1]), dim=-1
            ).values
            mean_margin = (
                top_probabilities[:, 0] - top_probabilities[:, 1]
                if top_probabilities.shape[-1] > 1
                else top_probabilities[:, 0]
            )
            outputs["tta_logs"] = {
                "centroid_mean_confidence": top_probabilities[:, 0].mean().item(),
                "centroid_mean_margin": mean_margin.mean().item(),
            }
            return outputs
