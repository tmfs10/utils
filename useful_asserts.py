
def assert_sorted(x, asc=True):
    if asc:
        for i in range(1, len(x)):
            assert x[i] >= x[i-1], "x[%d] = %s >= %s = x[%d]" % (i, str(x[i]), str(x[i-1]), i-1)
    else:
        for i in range(1, len(x)):
            assert x[i] <= x[i-1], "x[%d] = %s <= %s = x[%d]" % (i, str(x[i]), str(x[i-1]), i-1)
