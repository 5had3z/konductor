import pytest

from konductor.registry import Registry


def test_decorator():
    r = Registry("test")

    @r.register_module()
    class Foo:
        pass

    @r.register_module("bar")
    class Foo2:
        pass

    assert all(k in r for k in ["foo", "bar"]), "Missing registration"
    assert Foo2 == r["bar"] and Foo == r["foo"]

    with pytest.raises(KeyError):
        r["foo2"]
        r["Foo"]


def test_duplicate_keys():
    r = Registry("test")

    @r.register_module()
    class Foo:
        pass

    with pytest.raises(KeyError):

        @r.register_module("foo")
        class NewFoo:
            pass

    with pytest.raises(KeyError):

        @r.register_module()
        class foo:
            pass

    @r.register_module("foo", force_override=True)
    class NewFoo2:
        pass


def test_basic_functionality():
    r = Registry("test")
    assert r.name == "test", "Name does not match"
    r = Registry("Test")
    assert r.name == "test", "Forced lower isn't working"

    class Foo:
        pass

    r.register_module("foo", Foo)

    @r.register_module()
    class Bar:
        pass

    assert all(k in r for k in ["foo", "bar"])
    assert len(r) == 2
    rpr = (
        "Registry (name=test, items={"
        "'foo': <class 'tests.test_registry.test_basic_functionality.<locals>.Foo'>, "
        "'bar': <class 'tests.test_registry.test_basic_functionality.<locals>.Bar'>})"
    )
    assert r.__repr__() == rpr
    assert r.module_dict == {"foo": Foo, "bar": Bar}


def test_invalid_objects():
    r = Registry("test")

    with pytest.raises(TypeError):
        r.register_module("foo", 213)

    with pytest.raises(TypeError):
        r.register_module("bar", "asefde")
