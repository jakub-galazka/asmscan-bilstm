import matplotlib.pyplot as plt

from .dir import makedir


def config_fig(xlabel: str = None, ylabel: str = None, ylim: tuple[float] = None, style: str = "seaborn-v0_8") -> None:
    plt.style.use(style)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if bool(ylim):
        plt.ylim(ylim)

def savefig(filepath: str, legend: bool = True) -> None:
    if legend:
        plt.legend()
    plt.savefig(makedir(filepath))
    plt.clf()
