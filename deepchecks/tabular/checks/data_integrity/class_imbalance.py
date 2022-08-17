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
import numpy as np

from deepchecks.core import CheckResult, ConditionCategory, ConditionResult
from deepchecks.core.errors import DatasetValidationError
from deepchecks.tabular import Context, SingleDatasetCheck
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
            value is a ratio of least frequent class and most frequent class
            data is a bar graph that displays the frequency of each class
        -------
        CheckResult
            ratio of least frequent class and most frequent class.
        """


        dataset = context.get_data_by_kind(dataset_kind).data
        dataset = select_from_dataframe(dataset, self.columns, self.ignore_columns)
        label_name = dataset.label_name

        #Find function that extracts least frequent value in column and most frequent
        #See how to display a graph of most frequent
        #Figure out testing error (ask contributor)
