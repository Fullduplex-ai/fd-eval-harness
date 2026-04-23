import pytest

from fd_eval.core.adapter import FDModelAdapter
from fd_eval.core.registry import get_adapter, get_task, list_adapters, list_tasks
from fd_eval.core.task import Task


def test_registry_list_tasks():
    tasks = list_tasks()
    assert "voice_activity_detection" in tasks
    assert "turn_taking_latency" in tasks


def test_registry_list_adapters():
    adapters = list_adapters()
    assert "moshi" in adapters
    assert "energy_vad" in adapters


def test_registry_get_task():
    task_cls = get_task("voice_activity_detection")
    assert issubclass(task_cls, Task)
    assert task_cls.__name__ == "VoiceActivityDetection"


def test_registry_get_adapter():
    adapter_cls = get_adapter("energy_vad")
    assert issubclass(adapter_cls, FDModelAdapter)
    assert adapter_cls.__name__ == "EnergyVADAdapter"


def test_registry_missing_task():
    with pytest.raises(KeyError):
        get_task("nonexistent_task")


def test_registry_missing_adapter():
    with pytest.raises(KeyError):
        get_adapter("nonexistent_adapter")
