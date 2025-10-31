import numpy as np
import matplotlib.pyplot as plt

def plot_fisher_ellipse(theta0, F, level=0.68, ax=None, label=None, color=None, lw=2.0):
    """
    Plot the Gaussian 2D confidence ellipse implied by Fisher matrix F at center theta0.
    level: confidence level in 2D (0.393, 0.683, 0.955, 0.997 correspond to k^2≈1, 2.30, 6.17, 11.8).
    """
    # Quantiles for 2 dof
    level_to_k2 = {0.393: 1.00, 0.683: 2.30, 0.955: 6.17, 0.997: 11.8}
    k2 = level_to_k2.get(level, 2.30)

    C = np.linalg.pinv(F, rcond=1e-12)        # parameter covariance
    vals, vecs = np.linalg.eigh(C)            # eigenvalues ascending
    t = np.linspace(0, 2*np.pi, 400)
    circle = np.vstack([np.cos(t), np.sin(t)])
    ellipse = vecs @ np.diag(np.sqrt(vals * k2)) @ circle

    if ax is None:
        fig, ax = plt.subplots()

    ax.plot(theta0[0] + ellipse[0], theta0[1] + ellipse[1],
            label=label, linewidth=lw)
    ax.scatter([theta0[0]], [theta0[1]], s=25)
    ax.set_xlabel(r"$\theta_1$")
    ax.set_ylabel(r"$\theta_2$")
    ax.set_aspect("equal", adjustable="box")
    if label:
        ax.legend(frameon=False)
    return ax

# Example use with your Fisher:
theta0 = np.array([1.0, 2.0])
F = F_num  # from ForecastKit
ax = plot_fisher_ellipse(theta0, F, level=0.683, label="68% ellipse")
plt.title("Fisher Gaussian approx (68%)")
plt.tight_layout()
plt.show()
