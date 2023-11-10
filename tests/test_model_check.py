import pytest
from comma_placement.comma_fixer import CommaFixer


@pytest.fixture()
def model():
    yield CommaFixer("just097/roberta-base-lora-comma-placement-r-16-alpha-32", "cpu")


def test_invariance(model):
    samples = ["One two three.", "Four five six."]
    predicted = [model.fix_commas(sample) for sample in samples]
    assert predicted == ["One, two, three.", "Four, five, six."]


def test_directional(model):
    samples = ["Coffee an apple a milk", "There are no commas here.", "However there is a comma here."]
    predicted = [model.fix_commas(sample) for sample in samples]
    assert predicted == ["Coffee, an apple, a milk,", "There are no commas here.", "However, there is a comma here."]


def test_long_sentence(model):
    sample = """As the sun dipped below the horizon painting the sky in shades of orange red and purple the tranquility of the evening was interrupted by the sudden appearance of a shooting star."""
    predicted = model.fix_commas(sample)
    assert (
        predicted
        == "As the sun dipped below the horizon, painting the sky in shades of orange, red and purple, the tranquility of the evening was interrupted by the sudden appearance of a shooting star."
    )


def test_fix_wrong(model):
    sample = "I, am a man."
    predicted = model.fix_commas(sample)
    assert predicted == "I am a man."


def test_remove_commas(model):
    sample = "One, two, three."
    assert model.remove_commas(sample) == "One two three."
