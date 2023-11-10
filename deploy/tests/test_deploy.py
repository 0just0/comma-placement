import requests
import json


def test_get_request():
    response = requests.get("http://0.0.0.0:8008/")
    assert response.status_code == 405


def test_post_request():
    data = json.dumps({"input_text": "Hello"})
    response = requests.post("http://0.0.0.0:8008/", data)
    assert response.status_code == 200


def test_fix_commas():
    data = json.dumps({"input_text": "One Two three."})
    response = requests.post("http://0.0.0.0:8008/", data)
    assert response.status_code == 200
    assert response.json()["text_with_commas"] == "One, Two, three."
