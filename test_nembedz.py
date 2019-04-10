from sketch import _make_batch

def test_make_batch():
    item = lambda i: ([i],[i],[i])

    b = _make_batch([item(i) for i in range(1,8)], 3)
    assert(b == [[item(1),item(2),item(3)],
                 [item(4),item(5),item(6)],
                 [item(7)]])

    b = _make_batch([item(i) for i in range(1,7)], 3)
    assert(b == [[item(1),item(2),item(3)],
                 [item(4),item(5),item(6)]])
