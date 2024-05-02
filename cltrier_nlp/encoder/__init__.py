"""

"""
import logging
import typing

import pydantic
import torch
import transformers

from .batch import EncodedBatch
from .pooler import EncoderPooler
from .. import functional

__all__ = ["EncodedBatch", "EncoderPooler"]


class EncoderArgs(pydantic.BaseModel):
    """

    """
    model: str = "prajjwal1/bert-tiny"
    layers: typing.List[int] = [-1]

    device: str = functional.neural.get_device()

    tokenizer: typing.Dict[str, str | int] = dict(
        max_length=512,
        truncation=True,
        return_offsets_mapping=True,
    )


class Encoder(torch.nn.Module):
    """

    """

    @functional.timeit
    def __init__(self, args: EncoderArgs = EncoderArgs()):
        """

        """
        super().__init__()

        self.args = args

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)
        self.model = transformers.AutoModel.from_pretrained(
            args.model, output_hidden_states=True
        ).to(self.args.device)

        logging.info(self)

    def __call__(self, batch: typing.List[str], unpad: bool = True) -> EncodedBatch:
        """

        """
        encoding, token = self.tokenize(batch)
        embeds: torch.Tensor = self.forward(
            torch.tensor(encoding["input_ids"], device=self.args.device).long(),
            torch.tensor(encoding["attention_mask"], device=self.args.device).short(),
        )

        return EncodedBatch(
            **{
                "embeds": embeds,
                "token": token,
                "unpad": unpad,
            }
            | encoding
        )

    def tokenize(
        self, batch: typing.List[str], padding: bool = True
    ) -> typing.Tuple[typing.Dict, typing.List[typing.List[str]]]:
        """

        """
        return (
            encoding := self.tokenizer(batch, padding=padding, **self.args.tokenizer),
            [self.ids_to_tokens(ids) for ids in encoding["input_ids"]],
        )

    def forward(self, ids: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        """

        """
        return (
            torch.stack(
                [self.model.forward(ids, masks).hidden_states[i] for i in self.args.layers]
            )
            .sum(0)
            .squeeze()
        )

    def ids_to_tokens(self, ids: torch.Tensor) -> typing.List[str]:
        """

        """
        return self.tokenizer.convert_ids_to_tokens(ids)

    def ids_to_sent(self, ids: torch.Tensor) -> str:
        """

        """
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    @property
    def dim(self) -> int:
        """

        """
        return self.model.config.to_dict()["hidden_size"]

    def __len__(self) -> int:
        """

        """
        return self.model.config.to_dict()["vocab_size"]

    def __repr__(self) -> str:
        """

        """
        return (
            f'> Encoder Name: {self.model.config.__dict__["_name_or_path"]}\n'
            f"  Memory Usage: {functional.neural.calculate_model_memory_usage(self.model)}"
        )
