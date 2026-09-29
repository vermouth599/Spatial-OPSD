import asyncio
import time
from typing import Any, Dict, List, Optional

from kdflow.trainer.data_processor import RolloutDataProcessor
from kdflow.utils.logging_utils import init_logger

logger = init_logger(__name__)


class RolloutManager:
    """Manage batched rollout generation and data processing."""

    def __init__(
        self,
        strategy,
        rollout_group,
        is_same_tokenizer: bool,
        generate_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.args = strategy.args
        self.rollout_group = rollout_group
        self.generate_kwargs = generate_kwargs
        self.max_concurrent = (
            self.args.rollout.rollout_engine_concurrency
            * self.rollout_group.num_actors
        )
        if self.max_concurrent <= 0:
            raise ValueError(
                f"max_concurrent must be positive, got {self.max_concurrent}"
            )

        self.data_processor = RolloutDataProcessor(strategy, is_same_tokenizer)
        self.image_key = self.data_processor.image_key

    def rollout(
        self,
        prompt_batch: List[Dict[str, Any]],
        global_step: int,
        mode: str = "train",
        **kwargs,
    ) -> tuple[List[dict], Dict[str, float]]:
        """Generate rollout samples and convert them into micro-batches."""
        if mode not in ("train", "eval"):
            raise ValueError(f"Unsupported rollout mode: {mode!r}")
        if not prompt_batch:
            return [], {}

        n_samples = self.args.rollout.n_samples_per_prompt if mode == "train" else 1
        if n_samples <= 0:
            raise ValueError(f"n_samples_per_prompt must be positive, got {n_samples}")

        should_sleep = self.args.train.enable_sleep
        if should_sleep:
            self.rollout_group.wakeup()

        try:
            stu_prompts = [
                item["stu_prompt"]
                for item in prompt_batch
                for _ in range(n_samples)
            ]
            tea_prompts = [
                item["tea_prompt"]
                for item in prompt_batch
                for _ in range(n_samples)
            ]
            labels = [
                item["label"] for item in prompt_batch for _ in range(n_samples)
            ]
            images = None
            if self.image_key:
                images = [
                    item.get("images")
                    for item in prompt_batch
                    for _ in range(n_samples)
                ]
            teacher_routing_keys = None
            if self.args.kd.multi_teacher_config:
                teacher_routing_keys = [
                    item.get("teacher_routing_key")
                    for item in prompt_batch
                    for _ in range(n_samples)
                ]

            sampling_params = kwargs or self.generate_kwargs
            if not sampling_params:
                raise ValueError("sampling_params must not be empty")

            outputs, timing_metrics = self._generate(
                stu_prompts,
                sampling_params,
                image_data=images,
            )
            micro_batches, rollout_metrics = self.data_processor.process(
                stu_prompts=stu_prompts,
                tea_prompts=tea_prompts,
                outputs=outputs,
                labels=labels,
                sampling_params=sampling_params,
                global_step=global_step,
                mode=mode,
                images=images,
                teacher_routing_keys=teacher_routing_keys,
            )
            rollout_metrics.update(timing_metrics)
            return micro_batches, rollout_metrics
        finally:
            if should_sleep:
                self.rollout_group.sleep()

    def _generate(
        self,
        prompts: List[str],
        sampling_params: Dict[str, Any],
        image_data: Optional[List] = None,
    ) -> tuple[List[Dict[str, Any]], Dict[str, float]]:
        """Run ordered concurrent generation requests."""
        if not prompts:
            return [], {}
        if image_data is not None and len(image_data) != len(prompts):
            raise ValueError("image_data and prompts must have the same length")

        max_concurrent = min(len(prompts), self.max_concurrent)
        try:
            return asyncio.run(
                self._async_generate(
                    prompts=prompts,
                    sampling_params=sampling_params,
                    max_concurrent=max_concurrent,
                    image_data=image_data,
                )
            )
        except Exception:
            try:
                engine_health = self.rollout_group.health_check()
            except Exception as health_error:
                engine_health = f"unavailable ({health_error!r})"
            logger.exception(
                "Rollout generation failed: prompts=%d, max_concurrent=%d, "
                "engine_health=%s",
                len(prompts),
                max_concurrent,
                engine_health,
            )
            raise

    async def _async_generate(
        self,
        prompts: List[str],
        sampling_params: Dict[str, Any],
        max_concurrent: int,
        image_data: Optional[List] = None,
    ) -> tuple[List[Dict[str, Any]], Dict[str, float]]:
        """Schedule single-sample generation requests concurrently."""
        import aiohttp

        semaphore = asyncio.Semaphore(max_concurrent)
        results = [None] * len(prompts)
        rollout_times = [0.0] * len(prompts)

        async def run_request(
            index: int, prompt: str, session: aiohttp.ClientSession
        ) -> None:
            try:
                async with semaphore:
                    start = time.perf_counter()
                    results[index] = await self.rollout_group.generate_one(
                        prompt=prompt,
                        sampling_params=sampling_params,
                        session=session,
                        image_data=(
                            image_data[index] if image_data is not None else None
                        ),
                    )
                    rollout_times[index] = time.perf_counter() - start
            except Exception as error:
                raise RuntimeError(
                    f"Rollout generation failed for request {index}"
                ) from error

        connector = aiohttp.TCPConnector(limit=max_concurrent)
        timeout = aiohttp.ClientTimeout(total=None, sock_read=None, sock_connect=60)
        async with aiohttp.ClientSession(
            connector=connector, timeout=timeout
        ) as session:
            await asyncio.gather(
                *(
                    run_request(index, prompt, session)
                    for index, prompt in enumerate(prompts)
                )
            )

        timing_metrics = {
            "timing/rollout_per_sample/mean": sum(rollout_times) / len(rollout_times),
            "timing/rollout_per_sample/max": max(rollout_times),
        }
        return results, timing_metrics
