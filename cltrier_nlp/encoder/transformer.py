import logging
import os
import typing

import pydantic
import torch
import transformers

from .. import util

os.environ["TOKENIZERS_PARALLELISM"] = "false"
transformers.logging.set_verbosity_error()


class TransformerEncoderArgs(pydantic.BaseModel):
    model: str = "prajjwal1/bert-tiny"
    layers: typing.List[int] = [-1]

    tokenizer: typing.Dict = dict(
        max_length=512,
        truncation=True,
        return_offsets_mapping=True,
    )


class TransformerEncoder(torch.nn.Module):

    @util.timeit
    def __init__(self, args: TransformerEncoderArgs = TransformerEncoderArgs()):
        super().__init__()

        self.args = args

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)
        self.model = transformers.AutoModel.from_pretrained(
            args.model, output_hidden_states=True
        ).to(util.get_device())

        logging.info(self)

    def __call__(self, batch: typing.List[str], remove_padding: bool = True) -> typing.Dict:
        encoding, token = self.tokenize(batch)
        embeds: torch.Tensor = self.forward(
            torch.tensor(encoding['input_ids'], device=util.get_device()).long(),
            torch.tensor(encoding['attention_mask'], device=util.get_device()).short(),
        )

        return {
            label: (
                [v[:n] for v, n in zip(value, torch.tensor(encoding['attention_mask']).sum(1))]
                if remove_padding
                else value
            )
            for label, value in [('embeds', embeds), ('token', token)]
        } | encoding

    def tokenize(
        self, batch: typing.List[str], padding: bool = True
    ) -> typing.Tuple[typing.Dict, typing.List[typing.List[str]]]:
        return (
            encoding := self.tokenizer(batch, padding=padding, **self.args.tokenizer),
            [self.ids_to_tokens(ids) for ids in encoding['input_ids']],
        )

    def forward(self, ids: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        return (
            torch.stack(
                [self.model.forward(ids, masks).hidden_states[i] for i in self.args.layers]
            )
            .sum(0)
            .squeeze()
        )

    def ids_to_tokens(self, ids: torch.Tensor) -> typing.List[str]:
        return self.tokenizer.convert_ids_to_tokens(ids)

    def ids_to_sent(self, ids: torch.Tensor) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    @property
    def dim(self) -> int:
        return self.model.config.to_dict()['hidden_size']

    def __len__(self) -> int:
        return self.model.config.to_dict()['vocab_size']

    def __repr__(self) -> str:
        return (
            f'> Encoder Name: {self.model.config.__dict__["_name_or_path"]}\n'
            f'  Memory Usage: {util.calculate_model_memory_usage(self.model)}'
        )
