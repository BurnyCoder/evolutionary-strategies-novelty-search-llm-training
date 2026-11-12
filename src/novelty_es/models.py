"""Pinned local inference, complete generation logging, and deadline-aware decoding.

Generation API: https://huggingface.co/docs/transformers/en/main_classes/text_generation
Chat templates: https://huggingface.co/docs/transformers/en/chat_templating
Embedding normalization: https://sbert.net/docs/package_reference/sentence_transformer/SentenceTransformer.html
No model code is executed remotely and trust_remote_code is disabled.
"""

from __future__ import annotations

import gc
import time
from dataclasses import asdict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

from .data import grade, make_prompt
from .runtime import BudgetExpired, EventLog, check_budget


class DeadlineStop(StoppingCriteria):
    """Interrupt decoding at the next token when a phase reaches its absolute limit."""

    def __init__(self, deadline: float):
        self.deadline = deadline

    def __call__(self, input_ids, scores, **kwargs):
        return time.time() >= self.deadline


class BatchLogStreamer:
    """Emit complete partial texts every decoding block and retain final response logs.

    The first put contains prompt token IDs; subsequent puts contain one token per
    sequence. The final response event is authoritative and includes token counts.
    """

    def __init__(self, tokenizer, log: EventLog, scope: dict, row_ids: list[str]):
        self.tokenizer, self.log, self.scope, self.row_ids = tokenizer, log, scope, row_ids
        self.tokens = [[] for _ in row_ids]
        self.prompt = True
        self.step = 0

    def put(self, value):
        """Retain all tokens; decode prefixes instead of corrupting split UTF-8 tokens."""
        if self.prompt:
            self.prompt = False
            return
        for buffer, token in zip(self.tokens, value.reshape(-1).tolist(), strict=True):
            buffer.append(token)
        self.step += 1
        if self.step % 16 == 0:
            self.log.emit(
                "partial_outputs",
                **self.scope,
                row_ids=self.row_ids,
                token_step=self.step,
                outputs=self.tokenizer.batch_decode(self.tokens, skip_special_tokens=True),
            )

    def end(self):
        """Final responses are logged once their exact generated token counts are known."""


class ModelRunner:
    """One pinned BF16 inference model; CPU MiniLM avoids competing with perturbation buffers."""

    def __init__(self, config: dict, device: str, log: EventLog):
        self.config, self.device, self.log = config, device, log
        self.tokenizer = AutoTokenizer.from_pretrained(
            config["model_id"], revision=config["model_revision"], local_files_only=True, trust_remote_code=False
        )
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                config["model_id"],
                revision=config["model_revision"],
                local_files_only=True,
                trust_remote_code=False,
                dtype=torch.bfloat16,
                attn_implementation="sdpa",
            )
            .to(device)
            .eval()
        )
        self.model.requires_grad_(False)
        self.embedder = None

    def load_embedder(self) -> None:
        """Load only when training/calibrating; evaluation needs no embedding model."""
        if self.embedder is None:
            from sentence_transformers import SentenceTransformer

            self.embedder = SentenceTransformer(
                self.config["embedding_id"],
                revision=self.config["embedding_revision"],
                device="cpu",
                local_files_only=True,
            )
            self.embedder.eval()

    def embed(self, responses: list[str]) -> torch.Tensor:
        """Pass full probe strings through the frozen embedder's own truncation policy.

        The pinned MiniLM sentence_bert_config.json sets max_seq_length=256
        WordPiece tokens including special tokens; complete strings do not imply
        complete-token coverage. This historical policy is intentionally unchanged.
        https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/blob/1110a243fdf4706b3f48f1d95db1a4f5529b4d41/sentence_bert_config.json
        """
        self.load_embedder()
        return (
            self.embedder.encode(
                responses,
                batch_size=self.config["embedding_batch_size"],
                convert_to_tensor=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            .detach()
            .float()
            .cpu()
        )

    @torch.inference_mode()
    def generate(
        self, rows: list[dict], deadline: float, scope: dict, sampled: bool = False, sample_seed: int = 0
    ) -> list[dict]:
        """Generate a fixed batch and log full raw responses before exact grading."""
        check_budget(deadline)
        prompts = [make_prompt(row) for row in rows]
        chats = [
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for prompt in prompts
        ]
        for row, prompt, chat in zip(rows, prompts, chats, strict=True):
            self.log.emit("prompt", **scope, problem_id=row["id"], prompt=prompt, rendered_prompt=chat)
        inputs = self.tokenizer(chats, padding=True, return_tensors="pt").to(self.device)
        options = {
            "do_sample": sampled,
            "max_new_tokens": self.config["max_new_tokens"],
            "pad_token_id": self.tokenizer.pad_token_id,
            "use_cache": True,
            "stopping_criteria": StoppingCriteriaList([DeadlineStop(deadline)]),
            "streamer": BatchLogStreamer(self.tokenizer, self.log, scope, [r["id"] for r in rows]),
        }
        if sampled:
            # Explicitly disable model-card top-k/top-p defaults for the registered sampler.
            options.update(temperature=self.config["sampled_temperature"], top_p=1.0, top_k=0)
        torch.manual_seed(sample_seed)
        start = time.perf_counter()
        output = self.model.generate(**inputs, **options)
        tokens = output[:, inputs.input_ids.shape[1] :].cpu()
        texts = self.tokenizer.batch_decode(tokens, skip_special_tokens=True)
        elapsed = time.perf_counter() - start
        eos = self.model.generation_config.eos_token_id
        eos_ids = {eos} if isinstance(eos, int) else set(eos or [self.tokenizer.eos_token_id])
        records = []
        for row, prompt, text, token_row in zip(rows, prompts, texts, tokens.tolist(), strict=True):
            end = next((index + 1 for index, token in enumerate(token_row) if token in eos_ids), len(token_row))
            record = {
                "problem_id": row["id"],
                "prompt": prompt,
                "response": text,
                "numbers": row["numbers"],
                "target": row["target"],
                "generated_tokens": end,
                "token_ids": token_row[:end],
                "truncated": end == self.config["max_new_tokens"] and token_row[end - 1] not in eos_ids,
                "batch_seconds": elapsed,
                **asdict(grade(text, row["numbers"], row["target"])),
            }
            self.log.emit("response", **scope, **record)
            records.append(record)
        if time.time() >= deadline:
            raise BudgetExpired("Generation hit its phase deadline; incomplete batch excluded")
        return records

    def close(self) -> None:
        """Free model references between conditions so original weights are reloaded."""
        self.model = None
        self.embedder = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
