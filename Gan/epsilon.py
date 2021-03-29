import matplotlib.pyplot as plt

from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy

epsi = []
for epoch in range(10, 5000 + 1, 100):
    epsilon = (compute_dp_sgd_privacy.compute_dp_sgd_privacy(n=447,
                                                             batch_size=10,
                                                             noise_multiplier=3,
                                                             epochs=epoch,
                                                             delta=0.00223))
    epsi.append(epsilon[0])
epoch = []
for ep in range(1, 5000 + 1, 100):
    epoch.append(ep)

plt.plot(epoch, epsi)
plt.xlabel('Epochs')
plt.ylabel('Epsilon / Privacy Budget')
plt.title('Delta = 0.00223 (1/447)')
plt.legend(["Epsilon", "RDP-order"])
plt.show()
