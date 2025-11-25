import matplotlib.pyplot as plt

alphas = [0.5, 1.0, 1.5, 2.0, 2.5]

uiqm = [
    0.4358065991205419,
    0.4637773242695776,
    0.4799894355410205,
    0.4754802109391851,
    0.4436900348988911,
]

uciqe = [
    24.763173934267254,
    24.27763142422779,
    24.35887505220004,
    24.246864242213114,
    23.31892691247123,
]

contrast_gain = [
    1.0596842728050446,
    1.1273852023774562,
    1.1769441371204798,
    1.177527348526326,
    1.0868387921385645,
]

fig, axes = plt.subplots(1, 3, figsize=(12, 3), sharey=False)

axes[0].plot(alphas, uiqm, marker='o', color='r')
axes[0].set_xlabel('alpha')
axes[0].set_ylabel('UIQM')
axes[0].set_title('UIQM vs alpha')
axes[0].grid(True)

axes[1].plot(alphas, uciqe, marker='s', color='b')
axes[1].set_xlabel('alpha')
axes[1].set_ylabel('UCIQE')
axes[1].set_title('UCIQE vs alpha')
axes[1].grid(True)

axes[2].plot(alphas, contrast_gain, marker='^', color='g')
axes[2].set_xlabel('alpha')
axes[2].set_ylabel('Contrast Gain')
axes[2].set_title('Contrast Gain vs alpha')
axes[2].grid(True)

plt.tight_layout()
plt.show()
