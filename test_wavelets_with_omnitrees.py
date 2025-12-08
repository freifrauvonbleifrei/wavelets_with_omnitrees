from wavelets_with_omnitrees import get_numbers_with_ith_bit_set


def test_get_numbers_with_ith_bit_set():
    assert list(get_numbers_with_ith_bit_set(0, 3)) == [4, 5, 6, 7]
    assert list(get_numbers_with_ith_bit_set(1, 3)) == [2, 3, 6, 7]
    assert list(get_numbers_with_ith_bit_set(2, 3)) == [1, 3, 5, 7]
