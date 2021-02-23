def binary_search(a, key):
    # TODO: Write - Your - Code
    start = 0
    finish = len(a)

    while start <= finish:
        idx = (finish - start) // 2 + start
        if a[idx] > key:
            finish = idx
        elif a[idx] < key:
            start = idx
        else:
            return idx
    return -1

print(binary_search([1, 2, 3, 40, 47, 50, 60, 77], 55))
