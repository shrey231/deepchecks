# ----------------------------------------------------------------------------
# Copyright (C) 2021-2022 Deepchecks (https://www.deepchecks.com)
#
# This file is part of Deepchecks.
# Deepchecks is distributed under the terms of the GNU Affero General
# Public License (version 3 or later).
# You should have received a copy of the GNU Affero General Public License
# along with Deepchecks.  If not, see <http://www.gnu.org/licenses/>.
# ----------------------------------------------------------------------------
#

"""Module contains class_imbalance check."""
from typing import List, Union

import pandas as pd
import plotly.express as px

from deepchecks.core import CheckResult, ConditionCategory, ConditionResult
from deepchecks.core.errors import DatasetValidationError
from deepchecks.tabular import Context, SingleDatasetCheck
from deepchecks.tabular.dataset import _get_dataset_docs_tag
from deepchecks.utils.dataframes import select_from_dataframe
from deepchecks.utils.strings import format_list, format_percent
from deepchecks.utils.typing import Hashable

__all__ = ['ClassImbalance']

class ClassImbalance(SingleDatasetCheck):
    """Checks if class ratio (ratio between least and most prevelant class) is below a threshold.

    Parameters
    ----------
    columns : Union[Hashable, List[Hashable]] , default: None
        List of columns to check, if none given checks
        all columns Except ignored ones.
    ignore_columns : Union[Hashable, List[Hashable]] , default: None
        List of columns to ignore, if none given checks
        based on columns variable.
    n_to_show : int , default: 5
        number of most common duplicated samples to show.
    threshold: float, default: 0.1
        Condition to see if ratio between least prevelant class and most prevelant class is below this threshold
    """

    def __init__(
        self,
        columns: Union[Hashable, List[Hashable], None] = None,
        ignore_columns: Union[Hashable, List[Hashable], None] = None,
        n_to_show: int = 5,
        threshold: float=0.1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.columns = columns
        self.ignore_columns = ignore_columns
        self.threshold = threshold
        self.n_to_show = n_to_show

    def run_logic(self, context: Context, dataset_kind):
        """Run check.

        Returns
            value is the ratio that 
            data is a bar graph that displays the frequency of each class
        -------
        CheckResult
            ratio of least frequent class and most frequent class.
        """


        dataset = context.get_data_by_kind(dataset_kind).data
        dataset = select_from_dataframe(dataset, self.columns, self.ignore_columns)

        if len(dataset.columns) == 1:
            raise DatasetValidationError(
                'Dataset does not contain an index or a datetime',
                html=f'Dataset does not contain an index or a datetime. see {_get_dataset_docs_tag()}'
            )

        label_name = dataset.label_name


        most_frequent_class_name = dataset[label_name].value_counts().idxmax()
        least_frequent_class_name = dataset[label_name].value_counts().idxmin()
        all_class_frequency = dataset[label_name].value_counts().to_dict()

        class_ratio = round( all_class_frequency[least_frequent_class_name] / all_class_frequency[most_frequent_class_name], 2)

        if context.with_display:
            dataframe = pd.DataFrame(all_class_frequency)
            text = [f'A graph that shows the frequency of each class within the provided dataset.']

            xaxis_layout = dict(
                title='Classes',
                type='category',
               
                range=(-3, len(dataframe.index) + 2)
            )
            yaxis_layout = dict(
                fixedrange=True,
                range=(0, float('inf')),
                title='Class Frequency'
            )

            figure = px.bar(dataframe, x=dataframe.index, y='frequency')
            figure.update_layout(
                height=400
            )
            figure.update_layout(
                dict(
                    xaxis=xaxis_layout,
                    yaxis=yaxis_layout,
                    coloraxis=dict(
                        cmin=0,
                        cmax=1
                    )
                )
            )

            display = [figure, *text]
        else:
            display = None

        return CheckResult(value=class_ratio, display=display)

       


    def add_condition_ratio_less_than(self, threshold):
        """Add condition - require class imbalance ratio to be less than to the threshold.
         Parameters
        ----------
        threshold : int , default: None
            Delegates the ratio threshold that the class_imbalance ratio should surpass 
        """
        def max_ratio_condition(result: float) -> ConditionResult:
            details = f'Found the class ratio to be {result}'
            category = ConditionCategory.PASS if result < threshold else ConditionCategory.WARN
            return ConditionResult(category, details)

        return self.add_condition(f'Class Imbalance ratio is less than {format_percent(threshold)}',
                                  max_ratio_condition)