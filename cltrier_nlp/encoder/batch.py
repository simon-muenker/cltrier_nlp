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

    unpad: bool = True

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, __context) -> None:

        if self.unpad:
            mask = torch.tensor(self.attention_mask).sum(1)

            self.embeds = [v[:n] for v, n in zip(self.embeds, mask)]
            self.token = [v[:n] for v, n in zip(self.token, mask)]
