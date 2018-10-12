import numpy as np
import matplotlib.pyplot as plt


def threshold_plots(y_test, thresholds):
    fig, axs = plt.subplots(3,1, figsize=(6,16))

    #thresholds come from predict_proba

    accs = np.empty(y_test.shape)
    fprs = np.empty(y_test.shape)
    tprs = np.empty(y_test.shape)

    for i,t in enumerate(thresholds):
        pred_y = thresholds > t
        accs[i] = np.mean(pred_y == y_test)
        fprs[i] = np.mean(pred_y[y_test == 0])
        tprs[i] = np.mean(pred_y[y_test == 1])

    for ax, arr, name in zip(axs.flatten(), 
                       [accs, fprs, tprs],
                       ['accuracy', 'FPR', 'TPR']):
        ax.plot(sorted(log_pred_prob_test), arr)
        ax.set_xlabel("threshold")
        ax.set_ylabel(name)

    t_max_acc = thresholds[accs.argmax()]
    axs[0].vlines(t_max_acc, 0, 1, color='k', linestyle=':')

    plt.tight_layout()
    plt.show()
    print("Max accuracy is {:.2} at threshold {:.2}".format(accs.max(),     t_max_acc))
    print("Class balance is {:.2}".format(y.mean()))


def plot_roc(fprs, tprs, thresholds):
    plt.plot(fprs, tprs)
    plt.plot([0,1],[0,1], 'k:')
    plt.title("ROC Curve")
    plt.xlabel("FPR")
    plt.ylabel("TPR");