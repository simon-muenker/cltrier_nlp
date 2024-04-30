import typing

import torch

from . import transformer

POOL_FORM_FNS: typing.Dict[str, typing.Callable] = {
    # sentence based
    "sent_cls": lambda x: x[0],
    "sent_mean": lambda x: torch.mean(x[1:-1], dim=0),
    # word based, positional extraction
    "subword_first": lambda x: x[0],
    "subword_last": lambda x: x[-1],
    # word based, arithmetic extraction
    "subword_mean": lambda x: torch.mean(x, dim=0),
    "subword_min": lambda x: torch.min(x, dim=0)[0],
    "subword_max": lambda x: torch.max(x, dim=0)[0],
}

POOL_FORM_TYPE = typing.Literal[
    "sent_cls",
    "sent_mean",
    "subword_first",
    "subword_last",
    "subword_mean",
    "subword_min",
    "subword_max",
]


class TransformerEncoderPooler:

    def __call__(
        self,
        encodes: transformer.TransformerEncoderBatch,
        extract_spans: typing.List[typing.Tuple[int, int]] = None,
        form: POOL_FORM_TYPE = "sent_cls",
    ) -> typing.List[torch.Tensor]:
        return [
            POOL_FORM_FNS[form](embed)
            for embed in (
                encodes.embeds
                if form in ["sent_cls", "sent_mean"]
                else TransformerEncoderPooler._extract_embed_spans(encodes, extract_spans)
            )
        ]

    @staticmethod
    def _extract_embed_spans(
        encodes: transformer.TransformerEncoderBatch, extract_spans
    ) -> typing.Generator:
        for span, mapping, embeds in zip(extract_spans, encodes.offset_mapping, encodes.embeds):
            emb_span_idx = TransformerEncoderPooler._get_token_idx(
                mapping[1 : embeds.size(dim=0) - 1], span
            )
            yield embeds[emb_span_idx[0] : emb_span_idx[1] + 1]

    @staticmethod
    def _get_token_idx(
        mapping: typing.List[typing.Tuple[int, int]], c_span: typing.Tuple[int, int]
    ) -> typing.Tuple[int, int]:
        def prep_map(pos):
            return list(enumerate(list(zip(*mapping))[pos]))

        span: typing.Tuple[int, int] = (
            next(eid for eid, cid in reversed(prep_map(0)) if cid <= c_span[0]),
            next(eid for eid, cid in prep_map(1) if cid >= c_span[1]),
        )

        return span if span[0] <= span[1] else (span[1], span[0])
