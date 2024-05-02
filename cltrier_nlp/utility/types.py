import typing

_generic = typing.TypeVar('_generic')

Batch: typing.TypeAlias = typing.List[_generic]

Tokens: typing.TypeAlias = typing.List[str]
NGrams: typing.TypeAlias = typing.List[typing.Tuple[str, ...]]

