import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stat
from statsmodels.stats.weightstats import ztest

from lab1 import mean, std_dev, lambda_exp


def main():
    size = [25, 60, 150]

    norm_ = [stat.norm.rvs(loc=mean, scale=std_dev, size=s) for s in size]
    exp_ = [stat.expon.rvs(loc=0, scale=lambda_exp, size=s) for s in size]

    # check skew and kurtosis for normal 25 and 60

    print(f'Skew test -> {size[0]} : {stat.skew(norm_[0])} ; {size[1]} : {stat.skew(norm_[1])}')
    print(f'\nKurt test -> {size[0]} : {stat.kurtosis(norm_[0])} ; {size[1]} : {stat.kurtosis(norm_[1])}')

    # check mean intervals for normal 25 and 60

    print(f'\nMean int -> {size[0]} : {stat.norm.interval(confidence=0.95, loc=norm_[0].mean(), scale=norm_[0].std())} '
          f'; {size[1]} : {stat.norm.interval(confidence=0.95, loc=norm_[1].mean(), scale=norm_[1].std())} ')

    # F-test for normals 25 and 60
    print(stat.f_oneway(*norm_[:2]))

    # T-test for normals 25 and 60
    print(stat.ttest_ind(norm_[0], norm_[1]))

    #  Bartlett's test for exponential 60 and 150
    print(stat.bartlett(*exp_[1:]))

    # Confidential intervals for exponential
    print(f'\nIntervals mean:')
    for perc in [1, 5, 10]:
        for idx, arr in enumerate(exp_[1:]):
            print(f'\t{idx} {perc} % : {stat.expon.interval(confidence=(1 - perc / 100), scale=stat.sem(arr))}')

    print(f'\nIntegrals lambda:')
    for idx, arr in enumerate(exp_[1:]):
        print(f'\t{idx} {stat.expon.fit(arr)}')

    print(f'\nZ-test: {ztest(exp_[1], exp_[2], value=(exp_[1].mean() - exp_[2].mean()))}')


if __name__ == "__main__":
    main()
