import os
from typing import Callable, Optional, Dict, Any, List

from torch.utils.data import Dataset
from datasets import Image as ImageFeature
from PIL import Image

from kdflow.datasets.utils import (
    convert_to_openai_messages,
    get_tokenizer_or_processor,
    validate_dataset_columns,
)
from kdflow.models.utils import TokenizerCompareResult


class PromptDataset(Dataset):
    """
    Dataset for On-Policy Distillation

    Args:
        dataset: dataset for on-policy distillation
        strategy: training strategy object
        tokenizer_info: result of tokenizer comparison (template_identical, vocab_identical)
        max_data_num: maximum number of data to load
        input_template: optional template for formatting input
        num_processors: number of processors for parallel data loading
    """

    def __init__(
        self,
        dataset,
        strategy,
        tokenizer_info,
        max_data_num: int = None,
        input_template: Optional[str] = None,
        num_processors: int = 8,
    ) -> None:
        super().__init__()
        self.args = strategy.args
        self.strategy = strategy
        self.tokenizer_info = tokenizer_info or TokenizerCompareResult()
        self.template_identical = self.tokenizer_info.template_identical
        self.vocab_identical = self.tokenizer_info.vocab_identical
        self.same_tokenizer = self.tokenizer_info.is_identical
        self.strategy = strategy
        self.input_template = input_template
        
        # Config from strategy
        self.input_key = getattr(self.args.data, "input_key", None)
        self.teacher_input_key = getattr(self.args.data, "teacher_input_key", None) or self.input_key
        self.label_key = getattr(self.args.data, "label_key", None)
        self.apply_chat_template = getattr(self.args.data, "apply_chat_template", False)
        self.enable_thinking = getattr(self.args.model, "enable_thinking", False)
        self.prompt_max_len = getattr(self.args.data, "prompt_max_len", 0)

        self.image_key = getattr(self.args.data, "image_key", None)

        # Validate that all required columns exist in the dataset
        required_columns = {"input_key": self.input_key}
        if self.teacher_input_key != self.input_key:
            required_columns["teacher_input_key"] = self.teacher_input_key
        if self.label_key:
            required_columns["label_key"] = self.label_key
        if self.image_key:
            required_columns["image_key"] = self.image_key
        if self.args.kd.multi_teacher_config:
            required_columns["teacher_routing_key"] = self.args.data.teacher_routing_key
        validate_dataset_columns(dataset, **required_columns)

        # Load processor if multimodal
        self.student_processor = get_tokenizer_or_processor(
            self.args.model.student_name_or_path,
            need_processor=self.image_key is not None,
        )
        self.teacher_processors = {}
        if self.args.kd.multi_teacher_config:
            for teacher_key, teacher_path in self.args.kd.multi_teacher_config.items():
                self.teacher_processors[teacher_key] = get_tokenizer_or_processor(
                    teacher_path, need_processor=self.image_key is not None,
                )
        elif self.args.model.teacher_name_or_path is not None:
            self.teacher_processors["default"] = get_tokenizer_or_processor(
                self.args.model.teacher_name_or_path,
                need_processor=self.image_key is not None,
            )

        # Truncate dataset if max_data_num is specified
        if max_data_num is not None and max_data_num > 0 and max_data_num < len(dataset):
            strategy.log(f"Truncating dataset from {len(dataset)} to {max_data_num}")
            dataset = dataset.select(range(max_data_num))

        # Adjust worker count based on dataset size to speed up preprocessing.
        num_processors = min(
            num_processors,
            max(1, len(dataset) // 2000),
        )
        self.processed_dataset = dataset.map(
            self.process_data,
            remove_columns=dataset.column_names,
            num_proc=num_processors,
            load_from_cache_file=False,
            desc="Processing data",
        )
        if self.image_key:
            self.processed_dataset = self.processed_dataset.cast_column("images", [ImageFeature()])

        # Filter by prompt_max_len
        original_len = len(self.processed_dataset)
        if self.prompt_max_len > 0:
            strategy.log(f"Before prompt_max_len filter: {len(self.processed_dataset)}")
            self.processed_dataset = self.processed_dataset.filter(
                lambda prompt_len: prompt_len <= self.prompt_max_len,
                input_columns=["prompt_len"],
                desc="Filtering overlang samples",
            )
            strategy.log(f"After prompt_max_len filter: {len(self.processed_dataset)}")
            if len(self.processed_dataset) < original_len:
                self.strategy.log(
                    f"Filtered {original_len - len(self.processed_dataset)} samples "
                    f"exceeding prompt_max_len={self.prompt_max_len} "
                    f"({original_len} -> {len(self.processed_dataset)})"
                )
                
        self._print_sample()

    def _print_sample(self) -> None:
        """Print sample data for debugging."""
        self.strategy.print(f"Total samples: {len(self.processed_dataset)}")
        self.strategy.print(f"Sample student prompt:\n{self.processed_dataset[0]['stu_prompt']}")
        if not self.template_identical or self.teacher_input_key != self.input_key:
            self.strategy.print(f"Sample teacher prompt:\n{self.processed_dataset[0]['tea_prompt']}")

    def process_data(self, data: Dict) -> Dict[str, Any]:
        """Process a single data sample."""
        # Build student prompt
        stu_prompt = self._build_prompt(data, self.student_processor, self.input_key)
        
        # Build teacher prompt
        if self.args.kd.multi_teacher_config:
            routing_key = self.args.data.teacher_routing_key
            assert routing_key is not None, "`--teacher_routing_key` must be specified when using multi_teacher_config"
            assert routing_key in data, f"Routing key '{routing_key}' not found in data"
            teacher_key = data[routing_key]
            if teacher_key not in self.teacher_processors:
                raise ValueError(
                    f"Teacher routing key '{teacher_key}' not found in multi_teacher_config. "
                    f"Available keys: {list(self.teacher_processors.keys())}."
                )
            teacher_processor = self.teacher_processors[teacher_key]
            tea_prompt = self._build_prompt(data, teacher_processor, self.teacher_input_key)
        elif self.same_tokenizer and self.input_key == self.teacher_input_key:
            tea_prompt = stu_prompt
        else:
            tea_prompt = self._build_prompt(data, self.teacher_processors.get("default", self.student_processor), self.teacher_input_key)
        
        # Compute prompt token length for filtering
        tokenizer = self.student_processor.tokenizer if hasattr(self.student_processor, "tokenizer") else self.student_processor
        prompt_len = len(tokenizer.encode(stu_prompt))

        result = {
            "stu_prompt": stu_prompt,
            "tea_prompt": tea_prompt,
            "prompt": stu_prompt,
            "prompt_len": prompt_len,
            "label": data.get(self.label_key, "") if self.label_key else "",
            "datasource": data.get("datasource", "default"),
        }
        # Keep image references during preprocessing; ImageFeature decodes lazily
        # when items are fetched by the dataloader.
        if self.image_key:
            result["images"] = self._load_images(data.get(self.image_key))
            
        # load teacher routing info of each data for multi-teacher distillation
        if self.args.kd.multi_teacher_config is not None:
            assert self.args.data.teacher_routing_key is not None, "`--teacher_routing_key` must be specified when using multi_teacher_config"
            assert self.args.data.teacher_routing_key in data, f"Routing key '{self.args.data.teacher_routing_key}' not found in data"
            result["teacher_routing_key"] = data[self.args.data.teacher_routing_key]
            
        return result
    
    def _build_prompt(self, data: Dict, processor_or_tokenizer, input_key: str) -> str:
        """Build prompt from data with optional chat template or input template.
        
        Args:
            data: The data dict containing input
            processor_or_tokenizer: The processor or tokenizer to use for apply_chat_template
            input_key: The key to extract input from data
            
        Returns:
            Formatted prompt string
        """
        if self.apply_chat_template:
            chat = convert_to_openai_messages(data[input_key], expand_image=self.image_key is not None)
            rollout_suffix = os.environ.get("KDFLOW_ROLLOUT_RESPONSE_SUFFIX", "").strip()
            if rollout_suffix:
                # Opt-in trajectory shaping for on-policy KD. Apply it to the
                # final user turn for both student and privileged teacher so
                # their token positions remain directly comparable.
                for message in reversed(chat):
                    if message.get("role") != "user":
                        continue
                    content = message.get("content")
                    if isinstance(content, str):
                        message["content"] = content.rstrip() + "\n\n" + rollout_suffix
                        break
                    # Multimodal conversion represents a user turn as a list
                    # of {type: text|image} parts.  Preserve image parts and
                    # append the shaping instruction to its final text part.
                    if isinstance(content, list):
                        for part in reversed(content):
                            if isinstance(part, dict) and part.get("type") == "text":
                                part["text"] = str(part.get("text", "")).rstrip() + "\n\n" + rollout_suffix
                                break
                        else:
                            content.append({"type": "text", "text": rollout_suffix})
                        break
            chat = self._adapt_chat_for_processor(chat, processor_or_tokenizer)
            while chat and chat[-1].get("role", "user") == "assistant":
                chat.pop()
            chat_template_kwargs = {}
            if "enable_thinking" in str(getattr(processor_or_tokenizer, "chat_template", "")):
                chat_template_kwargs["enable_thinking"] = self.enable_thinking
            return processor_or_tokenizer.apply_chat_template(
                chat,
                tokenize=False,
                add_generation_prompt=True,
                **chat_template_kwargs,
            )
        
        prompt = data[input_key]
        return self.input_template.format(prompt) if self.input_template else prompt

    @staticmethod
    def _adapt_chat_for_processor(chat: List[Dict[str, Any]], processor_or_tokenizer) -> List[Dict[str, Any]]:
        """Adapt legacy string image placeholders for processors with structured image templates."""
        if getattr(processor_or_tokenizer, "image_token", None) != "<IMG_CONTEXT>":
            return chat

        adapted = []
        for message in chat:
            content = message.get("content")
            if not isinstance(content, str) or "<image>" not in content:
                adapted.append(message)
                continue

            parts = content.split("<image>")
            structured_content = []
            for index, part in enumerate(parts):
                if index > 0:
                    structured_content.append({"type": "image"})
                if part:
                    structured_content.append({"type": "text", "text": part})
            adapted.append({**message, "content": structured_content})
        return adapted

    def __len__(self) -> int:
        return len(self.processed_dataset)

    def __getitem__(self, idx: int) -> Dict[str, str]:
        """Get item by index.
        
        Returns:
            Dict with keys: datasource, stu_prompt, tea_prompt, label, images (optional)
        """
        item = self.processed_dataset[idx]
        result = {
            "datasource": item["datasource"],
            "stu_prompt": item["stu_prompt"],
            "tea_prompt": item["tea_prompt"],
            "label": item["label"],
        }
        if "images" in item:
            result["images"] = self._resize_images(item["images"])
        if "teacher_routing_key" in item:
            result["teacher_routing_key"] = item["teacher_routing_key"]
        return result

    @staticmethod
    def _resize_images(images: list) -> list:
        max_pixels = int(os.environ.get("KDFLOW_MAX_IMAGE_PIXELS", "0") or 0)
        if max_pixels <= 0:
            return images

        resized = []
        resampling = getattr(Image, "Resampling", Image).BICUBIC
        for image in images:
            if isinstance(image, str):
                image = Image.open(image).convert("RGB")
            elif isinstance(image, Image.Image) and image.mode != "RGB":
                image = image.convert("RGB")

            if isinstance(image, Image.Image):
                width, height = image.size
                pixels = width * height
                if pixels > max_pixels:
                    scale = (max_pixels / pixels) ** 0.5
                    new_size = (
                        max(1, int(width * scale)),
                        max(1, int(height * scale)),
                    )
                    image = image.resize(new_size, resampling)
            resized.append(image)
        return resized

    @staticmethod
    def _load_images(image_content) -> list:
        """Normalize image references from various input formats. Always returns a list."""
        if image_content is None:
            return []
        if isinstance(image_content, Image.Image):
            return [image_content]
        if isinstance(image_content, str):
            return [image_content]
        if isinstance(image_content, list):
            result = []
            for img in image_content:
                if isinstance(img, Image.Image):
                    result.append(img)
                elif isinstance(img, str):
                    result.append(img)
            return result
        return []

    @staticmethod
    def collate_fn(batch: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Collate function that simply returns the list of dicts.
        
        DataLoader will pass a list of dicts, we just return it as-is
        since rollout method expects a list of dicts.
        """
        return batch
