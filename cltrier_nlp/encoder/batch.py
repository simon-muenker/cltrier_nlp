import typing

import pydantic
import torch

from .. import utility


class EncoderBatch(pydantic.BaseModel):
    """

    """
    embeds: typing.Union[torch.Tensor, utility.types.Batch[torch.Tensor]]
    token: utility.types.Batch[utility.types.Tokens]

    input_ids: utility.types.Batch[typing.List[int]]
    token_type_ids: utility.types.Batch[typing.List[int]]

    attention_mask: utility.types.Batch[typing.List[int]]
    offset_mapping: utility.types.Batch[typing.List[typing.Tuple[int, int]]]

    unpad: bool = True

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, __context) -> None:

        if self.unpad:
            mask = torch.tensor(self.attention_mask).sum(1)

            self.embeds = [v[:n] for v, n in zip(self.embeds, mask)]
            self.token = [v[:n] for v, n in zip(self.token, mask)]
