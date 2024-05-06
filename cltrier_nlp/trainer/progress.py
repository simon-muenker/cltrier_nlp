import datetime
import logging
import typing

import pydantic
import pandas as pd


class Progress(pydantic.BaseModel):
    epoch: typing.List[int] = []
    duration: typing.List[datetime.timedelta] = []

    loss_train: typing.List[float] = []
    loss_test: typing.List[float] = []

    f1_train: typing.List[float] = []
    f1_test: typing.List[float] = []

    metric_train: typing.List[dict] = []
    metric_test: typing.List[dict] = []

    def append_record(
            self,
            epoch: int,
            duration: datetime.timedelta,
            train_results: typing.Tuple[float, float, dict],
            test_results: typing.Tuple[float, float, dict]
    ):
        """
        Append a record of the epoch, duration, train results, and test results to their respective lists.

        Args:
            epoch (int): The epoch number.
            duration (datetime.timedelta): The duration of the training.
            train_results (typing.Tuple[float, float, dict]): The results of the training.
            test_results (typing.Tuple[float, float, dict]): The results of the testing.

        """
        self.epoch.append(epoch)
        self.duration.append(duration)

        self.loss_train.append(train_results[0])
        self.loss_test.append(test_results[0])

        self.f1_train.append(train_results[1])
        self.f1_test.append(test_results[1])

        self.metric_train.append(train_results[2])
        self.metric_test.append(test_results[2])

    @pydantic.computed_field  # type: ignore[misc]
    @property
    def max_record_id(self) -> int:
        """
        Returns the maximum record id as an integer.
        """
        if not self.f1_test:
            return -1

        return self.f1_test.index(max(self.f1_test))

    @pydantic.computed_field  # type: ignore[misc]
    @property
    def last_is_best(self) -> bool:
        """
        Returns a boolean indicating whether the last element in f1_test is the maximum.
        """
        if not self.f1_test:
            return False

        return self.f1_test[-1] == max(self.f1_test)

    def log(self) -> None:
        """
        Logs the training and testing metrics, including loss and F1 scores, along with the epoch number and duration.
        """
        logging.info((
            f'[@{self.epoch[-1]:03}]: \t'
            f'loss_train={self.loss_train[-1]:2.4f} \t'
            f'loss_test={self.loss_test[-1]:2.4f} \t'
            f'f1_train={self.f1_train[-1]:2.4f} \t'
            f'f1_test={self.f1_test[-1]:2.4f} \t'
            f'duration={self.duration[-1]}'
        ))

    def export(self, path: str) -> None:
        """
        Exports the record dump to a CSV file at the specified path.

        Args:
            path (str): The file path to export the CSV.
        """
        (
            pd.DataFrame
            .from_records(self.model_dump(
                exclude={'metric_train', 'metric_test'}
            ), index=['epoch'])
            .to_csv(f'{path}.csv')
        )
