import typing


class Map:

    def __init__(self, keys: typing.Set[str]) -> None:
        self.keys2ids: typing.Dict[str, int] = {key: idx for idx, key in enumerate(keys)}
        self.ids2keys: typing.Dict[int, str] = {idx: key for key, idx in self.keys2ids.items()}

    def get_ids(self, keys: typing.List[str]) -> typing.List[int]:
        return [self.keys2ids[key] for key in keys]

    def get_keys(self, ids: typing.List[int]) -> typing.List[str]:
        return [self.ids2keys[idx] for idx in ids]

    def __len__(self):
        return len(self.keys2ids)
