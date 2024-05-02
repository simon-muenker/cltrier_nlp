"""

"""
import logging
import typing

import pydantic
import torch
import transformers

from .batch import EncoderBatch
from .pooler import EncoderPooler
from .. import functional
from .. import utility

__all__ = ["EncoderBatch", "EncoderPooler"]


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
        """Initialize the encoder with the provided EncoderConfig.

        Args:
            args (EncoderArgs): The configuration for the encoder.
        """
        super().__init__()

        self.args = args

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)
        self.model = transformers.AutoModel.from_pretrained(
            args.model, output_hidden_states=True
        ).to(self.args.device)

        logging.info(self)

    def __call__(self, batch: utility.types.Batch[str], unpad: bool = False) -> EncoderBatch:
        """Tokenizes input batch and returns embeddings and tokens with optional padding removal.

        Args:
            batch (utility.types.Batch[str]): List of input strings to be tokenized.
            unpad (bool, optional): Whether to remove padding from embeddings and tokens. Defaults to True.

        Returns:
            EncoderBatch: A EncodedBatch object. See documentation for more.
        """
        return EncoderBatch(
            **(encoding := self.tokenizer(batch, padding=True, **self.args.tokenizer)),
            **{
                "embeds": self.forward(
                    torch.tensor(encoding["input_ids"], device=self.args.device).long(),
                    torch.tensor(encoding["attention_mask"], device=self.args.device).short(),
                ),
                "token": [self.ids_to_tokens(ids) for ids in encoding["input_ids"]],
                "unpad": unpad,
            }

        )

    def forward(self, ids: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        """
        Perform a forward pass through the model and return the aggregated hidden states.

        Args:
            ids (torch.Tensor): The input tensor for token ids.
            masks (torch.Tensor): The input tensor for attention masks.

        Returns:
            torch.Tensor: The aggregated hidden states obtained from the model's forward pass.
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
        Convert the input token IDs to a list of token strings using the internal tokenizer.

        Args:
            ids (torch.Tensor): The input token IDs to be converted to tokens.

        Returns:
            List[str]: A list of token strings corresponding to the input token IDs.
        """
        return self.tokenizer.convert_ids_to_tokens(ids)

    def ids_to_sent(self, ids: torch.Tensor) -> str:
        """
        Convert the input tensor of token IDs to a string using the internal tokenizers decode method.

        Args:
            ids (torch.Tensor): The input tensor of token IDs.

        Returns:
            str: The decoded string output.
        """
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    @property
    def dim(self) -> int:
        """
        Return the dimension of the model.
        """
        return self.model.config.to_dict()["hidden_size"]

    def __len__(self) -> int:
        """
        Return the length of the object based on the vocabulary size.
        """
        return self.model.config.to_dict()["vocab_size"]

    def __repr__(self) -> str:
        """
        Return a string representation of the encoder including memory usage.
        """
        return (
            f'> Encoder Name: {self.model.config.__dict__["_name_or_path"]}\n'
            f"  Memory Usage: {functional.neural.calculate_model_memory_usage(self.model)}"
        )
