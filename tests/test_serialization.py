"""Tests for vetinari/utils/serialization.py — dataclass_to_dict utility."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

import pytest

from vetinari.utils.serialization import dataclass_to_dict


class Color(Enum):
    RED = "red"
    BLUE = "blue"


class Status(Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"


@dataclass
class SimpleData:
    name: str
    count: int
    tags: list[str] = field(default_factory=list)


@dataclass
class WithEnum:
    status: Status
    label: str = ""


@dataclass
class WithDatetime:
    created_at: datetime = field(default_factory=lambda: datetime(2024, 1, 1, tzinfo=timezone.utc))
    label: str = ""


@dataclass
class Nested:
    child: SimpleData = field(default_factory=lambda: SimpleData(name="default", count=0))
    color: Color = Color.RED


@dataclass
class WithOptional:
    name: str = ""
    value: int | None = None


@dataclass
class WithNestedList:
    items: list[SimpleData] = field(default_factory=list)
    metadata: dict[str, str] = field(default_factory=dict)


class TestDataclassToDict:
    def test_simple_dataclass(self):
        obj = SimpleData(name="test", count=5, tags=["a", "b"])
        result = dataclass_to_dict(obj)
        assert result == {"name": "test", "count": 5, "tags": ["a", "b"]}

    def test_enum_conversion(self):
        obj = WithEnum(status=Status.ACTIVE, label="x")
        result = dataclass_to_dict(obj)
        assert result == {"status": "active", "label": "x"}

    def test_datetime_conversion(self):
        dt = datetime(2024, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
        obj = WithDatetime(created_at=dt, label="ts")
        result = dataclass_to_dict(obj)
        assert result["created_at"] == dt.isoformat()
        assert result["label"] == "ts"

    def test_nested_dataclass(self):
        child = SimpleData(name="inner", count=3)
        obj = Nested(child=child, color=Color.BLUE)
        result = dataclass_to_dict(obj)
        assert result == {
            "child": {"name": "inner", "count": 3, "tags": []},
            "color": "blue",
        }

    def test_optional_none(self):
        obj = WithOptional(name="test", value=None)
        result = dataclass_to_dict(obj)
        assert result == {"name": "test", "value": None}

    def test_nested_list_of_dataclasses(self):
        items = [SimpleData(name="a", count=1), SimpleData(name="b", count=2)]
        obj = WithNestedList(items=items, metadata={"k": "v"})
        result = dataclass_to_dict(obj)
        assert len(result["items"]) == 2
        assert result["items"][0] == {"name": "a", "count": 1, "tags": []}
        assert result["metadata"] == {"k": "v"}

    def test_non_dataclass_raises(self):
        with pytest.raises(TypeError, match="Expected a dataclass instance"):
            dataclass_to_dict("not a dataclass")

    def test_class_type_raises(self):
        with pytest.raises(TypeError, match="Expected a dataclass instance"):
            dataclass_to_dict(SimpleData)

    def test_empty_dataclass(self):
        obj = SimpleData(name="", count=0)
        result = dataclass_to_dict(obj)
        assert result == {"name": "", "count": 0, "tags": []}

    def test_tuple_field_converted_to_list(self):
        """Tuple fields must become lists so json.dumps() can serialize the output."""

        @dataclass
        class WithTuple:
            coords: tuple

        dt = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        obj = WithTuple(coords=(1, dt, "label"))
        result = dataclass_to_dict(obj)
        assert isinstance(result["coords"], list), "tuple must be converted to list"
        assert result["coords"] == [1, dt.isoformat(), "label"]
