import pytest

from fd_eval.core import Task


def test_valid_task():
    class MyTask(Task):
        version = "0.1.0"
        scoring_method = "algorithmic"

    task = MyTask()
    assert task.scoring_method == "algorithmic"


def test_missing_scoring_method():
    with pytest.raises(ValueError, match="must define 'scoring_method'"):

        class BadTask(Task):
            version = "0.1.0"


def test_invalid_scoring_method():
    with pytest.raises(ValueError, match="invalid scoring_method"):

        class BadTask(Task):
            version = "0.1.0"
            scoring_method = "magic"


def test_other_scoring_method_missing_detail():
    with pytest.raises(ValueError, match="failed to provide 'scoring_method_detail'"):

        class BadTask(Task):
            version = "0.1.0"
            scoring_method = "other"


def test_other_scoring_method_with_detail():
    class GoodTask(Task):
        version = "0.1.0"
        scoring_method = "other"
        scoring_method_detail = "proprietary-eval-v1"

    task = GoodTask()
    assert task.scoring_method == "other"
    assert task.scoring_method_detail == "proprietary-eval-v1"
