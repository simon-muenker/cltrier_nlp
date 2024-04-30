import typing

import pydantic
import torch


class EncodedBatch(pydantic.BaseModel):
    embeds: typing.List[torch.Tensor]
    token: typing.List[typing.List[str]]

    input_ids: typing.List[typing.List[int]]
    token_type_ids: typing.List[typing.List[int]]

    attention_mask: typing.List[typing.List[int]]
    offset_mapping: typing.List[typing.List[typing.Tuple[int, int]]]

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)
