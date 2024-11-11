# test_sample.py

def add(a, b):
    return a + b

def test_add():
    assert add(2, 3) == 5  # This should pass
    assert add(1, 1) == 2  # This should pass
