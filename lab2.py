import numpy as np
import scipy.stats as stat
from lab1 import mean, variance, lambda_exp


def main():
    size = [25, 60, 150]

    norm_ = [stat.norm.rvs(loc=mean, scale=variance, size=s) for s in size[:2]]

    print(stat.f_oneway(*norm_))
    print(stat.ttest_ind(norm_[0], norm_[1]))




if __name__ == "__main__":
    main()
