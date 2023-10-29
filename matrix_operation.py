import typing
import numpy


def matrix_multiple(matrix1: typing.Union[typing.List[typing.List[typing.Union[int, float]]], numpy.ndarray],
                    matrix2: typing.Union[typing.List[typing.List[typing.Union[int, float]]], numpy.ndarray],
                    dtype: numpy.dtype = numpy.int32) \
        -> numpy.ndarray:
    if isinstance(matrix1, list):
        matrix1 = numpy.asarray(matrix1, dtype=dtype)
    if isinstance(matrix2, list):
        matrix2 = numpy.asarray(matrix2, dtype=dtype)
    row1, common, col2 = matrix1.shape[0], matrix1.shape[1], matrix2.shape[1]

    result = numpy.zeros((row1, col2), dtype=dtype)

    for i in range(row1):
        for j in range(col2):
            for k in range(common):
                result[i, j] += matrix1[i, k] * matrix2[k, j]
    return result
