import typing

import pandas as pd
import pydantic
from sklearn import metrics as sk_metrics


class Metric(pydantic.BaseModel):
    golds: pd.Series = pydantic.Field(default_factory=lambda: pd.Series)
    preds: pd.Series = pydantic.Field(default_factory=lambda: pd.Series)

    decoding_fn: typing.Callable = lambda x: x

    def add_observations(self, golds: pd.Series, preds: pd.Series):
        """
        Adds new observations to the existing golds and preds Series.

        Args:
            golds (pd.Series): The new gold observations to add.
            preds (pd.Series): The new predicted observations to add.
        """
        self.golds = golds if self.golds.empty else pd.concat([self.golds, golds], ignore_index=True)
        self.preds = preds if self.preds.empty else pd.concat([self.preds, preds], ignore_index=True)

    def f_score(self, reduce: str = "weighted") -> float:
        """
        Calculate the F1 score for the given gold standard and predicted values.

        Args:
            reduce (str): The reduction method for the F1 score calculation. Defaults to "weighted".

        Returns:
            float: The F1 score.
        """
        return sk_metrics.f1_score(self.golds, self.preds, average=reduce, zero_division=0.0)

    def accuracy(self) -> float:
        """

        Returns:
            float: The accuracy score.
        """
        return sk_metrics.accuracy_score(self.golds, self.preds)

    def cross_tabulation(self) -> pd.DataFrame:
        """
        Returns a pandas DataFrame containing the cross-tabulation of gold and predicted values.
        """
        return pd.crosstab(
            self.golds.apply(self.decoding_fn),
            self.preds.apply(self.decoding_fn),
            rownames=['gold'],
            colnames=['pred']
        )

    def classification_report(self) -> pd.DataFrame:
        """
        Return a classification report as a pandas DataFrame.

        Returns:
            pd.DataFrame: The classification report as a pandas DataFrame.
        """
        return pd.DataFrame(
            sk_metrics.classification_report(
                self.golds.apply(self.decoding_fn),
                self.preds.apply(self.decoding_fn),
                zero_division=0.0,
                output_dict=True
            )
        ).T.drop('accuracy')

    def export(self, path: str) -> None:
        """
        Export the classification report and cross tabulation to CSV files.

        Args:
            path (str): The path where the CSV files will be exported.
        """
        self.classification_report().to_csv(f'{path}/metric.classification_report.csv')
        self.cross_tabulation().to_csv(f'{path}/metric.cross_tabulation.csv')

    @pydantic.computed_field  # type: ignore[misc]
    @property
    def data(self) -> typing.Dict[str, pd.Series]:
        return {'golds': self.golds, 'preds': self.preds}

    def __repr__(self) -> str:
        """
        Return a string representation of the classification report as a dictionary.
        """
        return self.classification_report().__repr__()
