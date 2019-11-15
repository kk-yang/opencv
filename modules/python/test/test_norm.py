#!/usr/bin/env python

from itertools import product
from functools import reduce

import cv2 as cv
import numpy as np

from tests_common import NewOpenCVTests


def norm_inf(x, y=None):
    def norm(vec):
        return np.linalg.norm(vec, np.inf)

    return norm(x) if y is None else norm(x-y)


def norm_l1(x, y=None):
    def norm(vec):
        return np.linalg.norm(vec, 1)

    return norm(x) if y is None else norm(x-y)


def norm_l2(x, y=None):
    def norm(vec):
        return np.linalg.norm(vec)

    return norm(x) if y is None else norm(x-y)


def norm_l2sqr(x, y=None):
    def norm(vec):
        return np.square(vec).sum()

    return norm(x) if y is None else norm(x-y)


def norm_hamming(x, y=None):
    def norm(vec):
        return sum(bin(i).count('1') for i in vec)

    return norm(x) if y is None else norm(np.bitwise_xor(x, y))


def norm_hamming2(x, y=None):
    def norm(vec):
        def element_norm(element):
            binary_str = bin(element).split('b')[-1]
            if len(binary_str) % 2 == 1:
                binary_str = '0' + binary_str
            gen = filter(lambda p: p != '00',
                         (binary_str[i:i+2]
                          for i in range(0, len(binary_str), 2)))
            return sum(1 for _ in gen)

        return sum(element_norm(element) for element in vec)

    return norm(x) if y is None else norm(np.bitwise_xor(x, y))


norm_type_under_test = {
    cv.NORM_INF: norm_inf,
    cv.NORM_L1: norm_l1,
    cv.NORM_L2: norm_l2,
    cv.NORM_L2SQR: norm_l2sqr,
    cv.NORM_HAMMING: norm_hamming,
    cv.NORM_HAMMING2: norm_hamming2
}

norm_name = {
    cv.NORM_INF: 'inf',
    cv.NORM_L1: 'L1',
    cv.NORM_L2: 'L2',
    cv.NORM_L2SQR: 'L2SQR',
    cv.NORM_HAMMING: 'Hamming',
    cv.NORM_HAMMING2: 'Hamming2'
}


def get_element_types(norm_type):
    if norm_type in (cv.NORM_HAMMING, cv.NORM_HAMMING2):
        return (np.uint8,)
    else:
        return (np.int16, np.int32, np.float32, np.float64)


def generate_vector(size, dtype):
    if np.issubdtype(dtype, np.integer):
        return np.random.randint(0, 100, size).astype(dtype)
    else:
        return 10. + 12.5 * np.random.randn(size).astype(dtype)


sizes = (1, 2, 3, 5, 7, 19, 21, 32)


class CVNormTests(NewOpenCVTests):

    def test_norm_for_one_array(self):
        for norm_type, norm in norm_type_under_test.items():
            element_types = get_element_types(norm_type)
            for size, element_type in product(sizes, element_types):
                array = generate_vector(size, element_type)
                expected = norm(array)
                actual = cv.norm(array, norm_type)
                self.assertAlmostEqual(
                    expected, actual, places=2,
                    msg='Array {0} and norm {1}'.format(array,
                                                        norm_name[norm_type])
                )

    def test_norm_for_two_arrays(self):
        for norm_type, norm in norm_type_under_test.items():
            element_types = get_element_types(norm_type)
            for size, element_type in product(sizes, element_types):
                first = generate_vector(size, element_type)
                second = generate_vector(size, element_type)
                expected = norm(first, second)
                actual = cv.norm(first, second, norm_type)
                self.assertAlmostEqual(expected, actual, places=2,
                                       msg='Arrays {0} {1} and norm {2}'.format(
                                           first, second, norm_name[norm_type]
                                       ))

    def test_norm_fails_for_wrong_type(self):
        for norm_type in (cv.NORM_HAMMING, cv.NORM_HAMMING2):
            with self.assertRaises(Exception,
                                   msg='Type is not checked {0}'.format(
                                       norm_name[norm_type]
                                   )):
                cv.norm(np.array([1, 2], dtype=np.int32), norm_type)

    def test_norm_fails_for_array_and_scalar(self):
        for norm_type in norm_type_under_test:
            with self.assertRaises(Exception,
                                   msg='Exception is not thrown for {0}'.format(
                                       norm_name[norm_type]
                                   )):
                cv.norm(np.array([1, 2], dtype=np.uint8), 123, norm_type)

    def test_norm_fails_for_scalar_and_array(self):
        for norm_type in norm_type_under_test:
            with self.assertRaises(Exception,
                                   msg='Exception is not thrown for {0}'.format(
                                       norm_name[norm_type]
                                   )):
                cv.norm(4, np.array([1, 2], dtype=np.uint8), norm_type)

    def test_norm_fails_for_array_and_norm_type_as_scalar(self):
        for norm_type in norm_type_under_test:
            with self.assertRaises(Exception,
                                   msg='Exception is not thrown for {0}'.format(
                                       norm_name[norm_type]
                                   )):
                cv.norm(np.array([3, 4, 5], dtype=np.uint8),
                        norm_type, normType=norm_type)


if __name__ == '__main__':
    NewOpenCVTests.bootstrap()
