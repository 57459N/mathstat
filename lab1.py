import matplotlib.pyplot as plt
import numpy
import statistics
from scipy.stats import skew, kurtosis
from scipy.stats import norm as normal, expon

import seaborn as sea

mean = 90
std_dev = 11
lambda_exp = 0.0111

funcs = [statistics.mean,
         statistics.median,
         statistics.mode,
         statistics.variance,
         statistics.stdev,
         skew,
         kurtosis]


def main():
    for size in [25, 60, 160, 300, 1000]:
        norm = numpy.array(numpy.random.normal(loc=mean, scale=std_dev, size=size))
        exp = numpy.array(numpy.random.exponential(scale=lambda_exp, size=size))
        print(f'\nSuze: {size}')
        print('\nNorm:')
        for func in funcs:
            print(f'\t{func.__name__}: {func(norm)}')

        print('\nExp:')
        for func in funcs:
            print(f'\t{func(exp)=}')

        if size in [300, 1000]:
            sea.set()
            plt.figure()

            norm_plt = sea.displot(norm)
            exp_plt = sea.displot(exp)

            plt.show()

        if size in [25, 60, 160]:
            for confidence in [0.01, 0.05, 0.1]:
                t_crit_norm = normal.ppf((1 - confidence) / 2, size - 1)
                norm_interval_min = statistics.mean(norm) - statistics.stdev(norm) * t_crit_norm / numpy.sqrt(size)
                norm_interval_max = statistics.mean(norm) + statistics.stdev(norm) * t_crit_norm / numpy.sqrt(size)

                t_crit_exp = expon.ppf((1 - confidence) / 2, size - 1)
                expon_interval_min = statistics.mean(exp) - statistics.stdev(exp) * t_crit_exp / numpy.sqrt(size)
                expon_interval_max = statistics.mean(exp) + statistics.stdev(exp) * t_crit_exp / numpy.sqrt(size)

                print(f'\t\t Confidence intervals {confidence * 100}% : {(norm_interval_min, norm_interval_max)=}'
                      f' {(expon_interval_min, expon_interval_max)=}')
            ...


if __name__ == '__main__':
    main()
