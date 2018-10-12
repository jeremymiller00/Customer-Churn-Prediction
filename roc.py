def calc_for_roc(y_test, thresholds):
    '''
    thresholds come from predict_proba

    '''
    accs = np.empty(y_test.shape)
    fprs = np.empty(y_test.shape)
    tprs = np.empty(y_test.shape)

    for i,t in enumerate(thresholds):
        pred_y = thresholds > t
        accs[i] = np.mean(pred_y == y_test)
        fprs[i] = np.mean(pred_y[y_test == 0])
        tprs[i] = np.mean(pred_y[y_test == 1])
    return accs, fprs, tprs

def plot_roc(fprs, tprs, thresholds):
    plt.plot(fprs, tprs)
    plt.plot([0,1],[0,1], 'k:')
    plt.title("ROC Curve")
    plt.xlabel("FPR")
    plt.ylabel("TPR");