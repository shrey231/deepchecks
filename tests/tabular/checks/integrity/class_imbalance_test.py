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
"""Tests for Class Imbalance check"""
from hamcrest import assert_that, equal_to, calling, close_to, greater_than, has_items, has_length, is_in, raises
import numpy as np
import pandas as pd

from deepchecks.tabular.checks.data_integrity.class_imbalance import ClassImbalance
from deepchecks.core.errors import DatasetValidationError, DeepchecksNotSupportedError, DeepchecksValueError
from deepchecks.tabular.dataset import Dataset
from tests.base.utils import equal_condition_result

def generate_dataframe_and_expected(least):
    np.random.seed(42)
    df = pd.DataFrame(np.random.randn(100, 3), columns=['x1', 'x2', 'x3'])
    df['label'] = "yes"
    for i in range(100 - least, 100):
        df.at[i, "no"]

    return df, round(least/100-least, 2)

def test_class_imbalance_basic():
    df, expected = generate_dataframe_and_expected(11)
    result = ClassImbalance().run(dataset=Dataset(df, label='label', datetime_name='x2', index_name='x3',
                                                              cat_features=[]))
    assert_that(result.value, equal_to(expected))
    assert_that(result.display, has_length(greater_than(0)))

    df, expected = generate_dataframe_and_expected(9)
    result = ClassImbalance().run(dataset=Dataset(df, label='label', datetime_name='x2', index_name='x3',
                                                              cat_features=[]))
    assert_that(result.value, equal_to(expected))
    assert_that(result.display, has_length(greater_than(0)))

def test_class_imbalance_basic_no_display():
    df, expected = generate_dataframe_and_expected(11)
    result = ClassImbalance().run(dataset=Dataset(df, label='label', datetime_name='x2', index_name='x3',
                                                              cat_features=[]), with_display=False)
    assert_that(result.value, equal_to(expected))
    assert_that(result.display, has_length(equal_to(0)))

    df, expected = generate_dataframe_and_expected(9)
    result = ClassImbalance().run(dataset=Dataset(df, label='label', datetime_name='x2', index_name='x3',
                                                              cat_features=[]), with_display=False)
    assert_that(result.value, equal_to(expected))
    assert_that(result.display, has_length(equal_to(0)))


def generate_dataframe_and_expected_multi(least):
    np.random.seed(42)
    df = pd.DataFrame(np.random.randn(100, 3), columns=['x1', 'x2', 'x3'])
    df['label'] = "yes"
    for i in range(0,25):
        df.at[i, "yes"]

    for i in range(25, 100 - least):
        df.at[i, "maybe"]
    
    for i in range(100 - least, 100):
        df.at[i, "no"]

    return df, round(least/max(25, 100 - least), 2)

def test_class_imbalance_multi_basic():
    df, expected = generate_dataframe_and_expected_multi(11)
    result = ClassImbalance().run(dataset=Dataset(df, label='label', datetime_name='x2', index_name='x3',
                                                              cat_features=[]))
    assert_that(result.value, equal_to(expected))
    assert_that(result.display, has_length(greater_than(0)))

    df, expected = generate_dataframe_and_expected_multi(9)
    result = ClassImbalance().run(dataset=Dataset(df, label='label', datetime_name='x2', index_name='x3',
                                                              cat_features=[]))
    assert_that(result.value, equal_to(expected))
    assert_that(result.display, has_length(greater_than(0)))

def test_class_imbalance_multi_no_display():
    df, expected = generate_dataframe_and_expected_multi(11)
    result = ClassImbalance().run(dataset=Dataset(df, label='label', datetime_name='x2', index_name='x3',
                                                              cat_features=[]), with_display=False)
    assert_that(result.value, equal_to(expected))
    assert_that(result.display, has_length(equal_to(0)))

    df, expected = generate_dataframe_and_expected_multi(9)
    result = ClassImbalance().run(dataset=Dataset(df, label='label', datetime_name='x2', index_name='x3',
                                                              cat_features=[]), with_display=False)
    assert_that(result.value, equal_to(expected))
    assert_that(result.display, has_length(equal_to(0)))

def test_condition_pass():
    df, expected = generate_dataframe_and_expected_multi(11)

    check = ClassImbalance().add_condition_ratio_less_than(0.1)

    # Act
    result = check.conditions_decision(check.run(Dataset(df, label='label', datetime_name='x2', index_name='x3',
                                                         cat_features=[])))

    assert_that(result, has_items(
        equal_condition_result(is_pass=True,
                               details='Found the class ratio to be 0.12',
                               name='Class Imbalance ratio is less than 0.1')
    ))


def test_condition_fail():
    df, expected = generate_dataframe_and_expected_multi(11)

    check = ClassImbalance().add_condition_ratio_less_than(0.5)

    # Act
    result = check.conditions_decision(check.run(Dataset(df, label='label', datetime_name='x2', index_name='x3',
                                                         cat_features=[])))

    assert_that(result, has_items(
        equal_condition_result(is_pass=False,
                               details='Found the class ratio to be 0.12',
                               name='Class Imbalance ratio is less than 0.5')
    ))


def test_dataset_no_label():
    df, _ = generate_dataframe_and_expected()
    df = Dataset(df)
    assert_that(
        calling(ClassImbalance().run).with_args(dataset=df),
        raises(DeepchecksNotSupportedError,
               'Dataset does not contain a label column')
    )


def test_dataset_only_label():
    df, _ = generate_dataframe_and_expected()
    df = Dataset(df, label='label')
    assert_that(
        calling(ClassImbalance().run).with_args(dataset=df),
        raises(
            DatasetValidationError,
            'Dataset does not contain an index or a datetime')
    )

def test_dataset_wrong_input():
    wrong = 'wrong_input'
    assert_that(
        calling(ClassImbalance().run).with_args(wrong),
        raises(
            DeepchecksValueError,
            'non-empty instance of Dataset or DataFrame was expected, instead got str')
    )