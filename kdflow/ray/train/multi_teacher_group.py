import json
from collections import defaultdict
from typing import Optional, Tuple, Union

import numpy as np
import ray
import torch
from ray.util.placement_group import PlacementGroup

from kdflow.ray.train.teacher_group import TeacherActorGroup
from kdflow.utils.logging_utils import init_logger


logger = init_logger(__name__)

class MultiTeacherActorGroup:
    """
    Manages multiple TeacherActorGroup instances for multi-teacher distillation.
    """
    
    def __init__(
        self,
        strategy,
        num_gpus: int,
        num_gpus_per_node: int = 8,
        num_gpus_per_actor: float = 0.2,
        pg: Optional[Union[PlacementGroup, Tuple[PlacementGroup, list, list]]] = None,
    ):
        """
        Create multiple TeacherActorGroup instances based on multi_teacher_config.
        
        Args:
            strategy: Training strategy containing configuration args
            num_gpus: Total number of GPUs available (e.g., 8)
            num_gpus_per_node: Number of GPUs per physical node
            num_gpus_per_actor: Ray GPU resources per actor (fractional for co-location)
            pg: 3-tuple (pg, reordered_bundle_indices, reordered_gpu_ids), PlacementGroup, or None
        """
        self.teacher_groups = {}
        multi_teacher_config = strategy.args.kd.multi_teacher_config
        for i, teacher_key in enumerate(multi_teacher_config.keys()):
            logger.info(
                f"[MultiTeacherActorGroup] [{i+1}/{len(multi_teacher_config)}] "
                f"Start to initialize [{teacher_key}] teacher..."
            )
            self.teacher_groups[teacher_key] = TeacherActorGroup(
                strategy, 
                num_gpus=num_gpus, 
                num_gpus_per_node=num_gpus_per_node, 
                num_gpus_per_actor=num_gpus_per_actor, 
                pg=pg, 
                teacher_name_or_path=multi_teacher_config[teacher_key]
            )
        logger.info("[MultiTeacherActorGroup] All teachers ready.")

    def forward(self, global_batch):
        """
        Route different data samples to the corresponding TeacherActorGroup.
        This needs to break the original global_batch and re-organize them according to `--teacher_routing_key`
        """
        routed_samples = defaultdict(list)
        routed_sample_ids = defaultdict(list)
        for mbid, micro_batch in enumerate(global_batch):
            for i, key in enumerate(micro_batch["teacher_routing_key"]):
                routed_samples[key].append(_slice_sample(micro_batch, i))
                routed_sample_ids[key].append((mbid, i))
        for key in routed_samples:
            if key not in self.teacher_groups:
                raise ValueError(
                    f"Teacher routing key '{key}' not found in multi_teacher_config. "
                    f"Available keys: {list(self.teacher_groups.keys())}. "
                    f"Please check your data's 'teacher_routing_key' field or --multi_teacher_config."
                ) 
            self.teacher_groups[key].wakeup()
            routed_samples[key] = self.teacher_groups[key].forward(routed_samples[key])
            self.teacher_groups[key].sleep()

        slots = [[None] * len(mb["teacher_routing_key"]) for mb in global_batch]
        for key, sample_ids in routed_sample_ids.items():
            for (mbid, i), result in zip(sample_ids, routed_samples[key]):
                v = result["teacher_hiddens"]
                if isinstance(v, ray.ObjectRef):
                    v = ray.get(v)
                slots[mbid][i] = v
        for mbid, mb in enumerate(global_batch):
            mb["teacher_hiddens"] = slots[mbid]

        return global_batch
    
    def sleep(self, tags=None):
        """Just pass here since sleep&wakeup is handled in forward."""
        pass

    def wakeup(self, tags=None):
        """Just pass here since sleep&wakeup is handled in forward."""
        pass

    def shutdown(self):
        """Shutdown all teacher engines."""
        for key in self.teacher_groups:
            self.teacher_groups[key].shutdown()
        logger.info("[MultiTeacherActorGroup] All teacher groups shutdown complete.")


def _slice_sample(mb: dict, i: int) -> dict:
    out = {}
    for k, v in mb.items():
        if isinstance(v, (torch.Tensor, list)) and getattr(v, "ndim", 1) >= 1:
            out[k] = v[i:i+1]
        else:
            out[k] = v
    return out