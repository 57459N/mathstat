from typing import Optional
import matplotlib.pyplot as plt
import scipy.stats as stat
from lab1 import mean, std_dev, lambda_exp
from scipy.stats import chisquare
import numpy as np


def get_amounts(values: np.array, freedom_degree: int):
    amounts = [0] * freedom_degree
    minimum = min(values)
    maximum = max(values)
    step = (maximum - minimum) / freedom_degree
    for el in values:
        idx = int((el - minimum) / step)
        try:
            amounts[idx] += 1
        except Exception as e:
            amounts[idx - 1] += 1

    return amounts


def show_diff_plot(first_data: np.array, second_data: np.array, title: Optional[str]):
    plt.figure()
    plt.title(title)
    size = len(first_data)
    plt.plot(range(size), first_data / max(first_data) * max(second_data))
    plt.plot(range(size), second_data)
    plt.show()


def main():
    size3 = 150
    size4 = 300
    freedom_degree = 20

    # Generating samples
    norm_val3 = stat.norm.rvs(loc=mean, scale=std_dev, size=size3)
    norm_val4 = stat.norm.rvs(loc=mean, scale=std_dev, size=size4)

    exp_val3 = stat.expon.rvs(loc=0, scale=lambda_exp, size=size3)
    exp_val4 = stat.expon.rvs(loc=0, scale=lambda_exp, size=size4)

    # Generating references
    norm_x = np.linspace(-3, 3, freedom_degree)
    norm_dens = stat.norm.pdf(x=norm_x)

    exp_x = np.linspace(0, 5, freedom_degree)
    exp_dens = stat.expon.pdf(x=exp_x)

    # Histograms
    norm_amounts3 = get_amounts(norm_val3, freedom_degree)
    norm_amounts4 = get_amounts(norm_val4, freedom_degree)

    exp_amounts3 = get_amounts(exp_val3, freedom_degree)
    exp_amounts4 = get_amounts(exp_val4, freedom_degree)

    # Differences plots
    show_diff_plot(norm_dens, norm_amounts3, f'Normal pdf {size3}')
    show_diff_plot(norm_dens, norm_amounts4, f'Normal pdf {size4}')

    show_diff_plot(exp_dens, exp_amounts3, f'Exponential pdf {size3}')
    show_diff_plot(exp_dens, exp_amounts4, f'Exponential pdf {size4}')

    # Normalize densities
    exp_dens /= sum(exp_dens)
    norm_dens /= sum(norm_dens)

    print('Chi^2 for')
    for i in [1, 5, 10]:
        print(f'\t100 - {i} % : {stat.chi2.ppf(1 - i / 100, df=freedom_degree)}')

    # Normal
    print(f'Chi^2 normal {size4}  : {chisquare(norm_amounts4, f_exp=norm_dens * size4)}')
    print(f"KS normal {size3} : {stat.kstest(norm_val3, 'norm', args=(norm_val3.mean(), norm_val3.std()))}")

    # Exponential
    print(f'Chi^2 exp {size4}  : {chisquare(exp_amounts4, f_exp=(exp_dens * size4))}')
    print(f"KS exp {size3} : {stat.kstest(exp_val3, 'expon', args=(0, exp_val3.mean()))}")


if __name__ == "__main__":
    main()
