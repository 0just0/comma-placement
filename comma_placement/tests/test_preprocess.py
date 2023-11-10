from comma_placement.utils import data_process


def test_remove_spaces():
    a = ["Hello , World !", "'Name ' "]
    assert data_process.remove_spaces(a) == ["Hello, World!", "'Name'"]


def test_remove_empty():
    samples = ["One, two, three.", "", "Hello."]
    assert data_process.remove_empty(samples) == ["One, two, three.", "Hello."]


def test_remove_unk():
    samples = ["Hello!", "Hello, <unk>!"]
    assert data_process.remove_unk(samples) == ["Hello!"]


def test_remove_short():
    samples = ["a" * 100, "Hello string", "a" * 350]
    assert data_process.remove_short(samples) == ["a" * 350]


def test_remove_titles():
    samples = ["One == Two", "Normal string."]
    assert data_process.remove_titles(samples) == ["Normal string."]
