from matplotlib import pyplot as plt


def plot_figure(rewards, steps, name):
    plt.figure(figsize=(13, 7))

    plt.subplot(211)
    plt.plot(rewards, color='blue')
    plt.title(f"Algorytm {name}", fontsize=20)
    plt.ylabel("Wyniki nagród", fontsize=17)
    plt.grid(True)
    plt.xlim(0, len(rewards))

    plt.subplot(212)
    plt.plot(steps, color='blue')
    plt.ylabel("Ilość kroków", fontsize=17)
    plt.xlabel("Epizody", fontsize=19)
    plt.grid(True)
    plt.xlim(0, len(rewards))

    plt.tight_layout()
    plt.savefig(f"{name}.png")
    plt.show()
