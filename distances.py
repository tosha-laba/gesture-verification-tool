import time
import numpy as np

from math import sqrt


def interpolate(h):
    """
    Линейно интерполирует гистограмму (необходимо для корректной обработки масштабированных гистограмм).
    """
    x, y = list(h.keys()), list(h.values())
    return np.interp(range(min(x), max(x) + 1), x, y)


def normalize(h):
    """
    Нормализует гистограмму.
    """
    m = sum(h)
    return [x / m for x in h]


def total_prob(probs):
    """
    Вычисляет полную вероятность по частным и условным вероятностям.
    """
    return sum(map(lambda v: v[0] * v[1], probs))


def make_decision(prob):
    """
    Выдает решение по вероятности.
    """
    if prob >= 0.75:
        return 'Жесты совпадают'
    elif prob <= 0.35:
        return 'Подобие слабое'
    else:
        return 'Не получен чёткий ответ'


def prepare_hists(h1, h2, do_normalization=False):
    """
    Подготавливает гистограммы к работе (интерполирует значения, нормализует).
    """
    hi1, hi2 = interpolate(h1), interpolate(h2)
    first = 0
    last = min(len(hi1), len(hi2)) - 1

    if do_normalization:
        hi1 = normalize(hi1)
        hi2 = normalize(hi2)

    return hi1, hi2, first, last


def correlation(h1, h2):
    """
    Вычисляет коэффициент корреляции гистограм 'h1', 'h2'.
    """
    start = time.time()

    hi1, hi2, first, last = prepare_hists(h1, h2)

    avg1 = sum(h1.values()) / len(h1)
    avg2 = sum(h2.values()) / len(h2)

    acc_u = 0
    acc_d1 = 0
    acc_d2 = 0
    for i in reversed(range(first, last + 1)):
        acc_u += (hi1[i] - avg1) * (hi2[i] - avg2)
        acc_d1 += (hi1[i] - avg1) ** 2
        acc_d2 += (hi2[i] - avg2) ** 2

    result = acc_u / sqrt(acc_d1 * acc_d2)
    end = time.time()

    return result, end - start


def chi_square(h1, h2):
    """
    Вычисляет критерий Хи-квадрат гистограмм 'h1', 'h2'.
    """
    start = time.time()

    hi1, hi2, first, last = prepare_hists(h1, h2, True)

    acc = 0
    for i in reversed(range(first, last + 1)):
        acc += ((hi1[i] - hi2[i]) ** 2) / hi1[i]

    end = time.time()

    return acc, end - start


def intersection(h1, h2):
    """
    Вычисляет меру по пересечению 'h1', 'h2'.
    """
    start = time.time()

    hi1, hi2, first, last = prepare_hists(h1, h2, True)

    acc = 0
    for i in reversed(range(first, last + 1)):
        acc += min(hi1[i], hi2[i])

    end = time.time()

    return acc, end - start


def bhattacharyya(h1, h2):
    """
    Вычисляет расстояние Бхаттачарии между 'h1', 'h2'.
    """
    start = time.time()

    hi1, hi2, first, last = prepare_hists(h1, h2)

    avg1 = sum(h1.values()) / len(h1)
    avg2 = sum(h2.values()) / len(h2)

    acc = 0
    for i in reversed(range(first, last + 1)):
        acc += sqrt(hi1[i] * hi2[i])

    result = sqrt(abs(1 - (1 / (sqrt(avg1 * avg2 * ((last - first) ** 2)))) * acc))
    end = time.time()

    return result, end - start
