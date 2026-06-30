#
#  Class RocCurveCrossVal for ploting ROC curves of k-fold validation in single plot
#
#   Based in Sklearn examples.
#
#   Usage:
#         roc_avg = RocCurveCrossVal()                          # init outside fold loop
#         roc_avg.add_fold_result(label_auc, y_hat_auc, k)      # add each fold results
#         roc_avg.roc_avg_plot("TITLE")                         # after loop call plot finally
#
#  DGPP 2021-07-02
#


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_curve

# define types of mean to display value in graph
GRAPH_MEAN = 0
CALC_MEAN = 1


def plot_roc_curve(y, y_pred, *, sample_weight=None,
                   drop_intermediate=True, response_method="auto",
                   name=None, ax=None, pos_label=None, **kwargs):

    fpr, tpr, _ = roc_curve(y, y_pred, pos_label=pos_label,
                            sample_weight=sample_weight,
                            drop_intermediate=drop_intermediate)
    roc_auc = auc(fpr, tpr)

    viz = RocCurveDisplay(
        fpr=fpr,
        tpr=tpr,
        roc_auc=roc_auc,
        estimator_name=name,
        pos_label=pos_label
    )

    return viz.plot(ax=ax, name=name, **kwargs)


# class for plot curves externally
class RocCurveCrossVal:
    """
    curve_type=GRAPH_MEAN (0) or CALC_MEAN (1) (this one is from np.mean(aucs))
    """
    def __init__(self, curve_type=GRAPH_MEAN, print_chance=False):

        self.tprs = []
        self.aucs = []
        self.mean_fpr = np.linspace(0, 1, 10000) # Big value for good approx. for AUC Mean
        self.fig, self.ax = plt.subplots()
        self.curve_type = curve_type
        self.print_chance = print_chance

    def add_fold_result(self, label_auc, y_hat_auc, k):
        viz = plot_roc_curve(label_auc.ravel(), y_hat_auc.ravel(),
                             name='ROC fold {}'.format(k),
                             alpha=0.3, lw=1, ax=self.ax)
        interp_tpr = np.interp(self.mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        self.tprs.append(interp_tpr)
        self.aucs.append(viz.roc_auc)

    def roc_avg_plot(self, title):
        if self.print_chance:
            self.ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
                         label='Chance', alpha=.8)

        mean_tpr = np.mean(self.tprs, axis=0)
        mean_tpr[-1] = 1.0
        if self.curve_type == GRAPH_MEAN:
            mean_auc = auc(self.mean_fpr, mean_tpr)
        elif self.curve_type == CALC_MEAN:
            mean_auc = np.mean(self.aucs)
        std_auc = np.std(self.aucs)
        self.ax.plot(self.mean_fpr, mean_tpr, color='b',
                     label=r'Mean ROC (AUC = %0.4f $\pm$ %0.4f)' % (mean_auc, std_auc),
                     lw=2, alpha=.8)

        std_tpr = np.std(self.tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        self.ax.fill_between(self.mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                             label=r'$\pm$ 1 std. dev.')

        self.ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
                    title=title)
        self.ax.legend(loc="lower right")
        plt.show()


class RocCurveDisplay:
    """ROC Curve visualization.

    It is recommend to use :func:`~sklearn.metrics.plot_roc_curve` to create a
    visualizer. All parameters are stored as attributes.

    Read more in the :ref:`User Guide <visualizations>`.

    Parameters
    ----------
    fpr : ndarray
        False positive rate.

    tpr : ndarray
        True positive rate.

    roc_auc : float, default=None
        Area under ROC curve. If None, the roc_auc score is not shown.

    estimator_name : str, default=None
        Name of estimator. If None, the estimator name is not shown.

    pos_label : str or int, default=None
        The class considered as the positive class when computing the roc auc
        metrics. By default, `estimators.classes_[1]` is considered
        as the positive class.

        .. versionadded:: 0.24

    Attributes
    ----------
    line_ : matplotlib Artist
        ROC Curve.

    ax_ : matplotlib Axes
        Axes with ROC Curve.

    figure_ : matplotlib Figure
        Figure containing the curve.

    See Also
    --------
    roc_curve : Compute Receiver operating characteristic (ROC) curve.
    plot_roc_curve : Plot Receiver operating characteristic (ROC) curve.
    roc_auc_score : Compute the area under the ROC curve.

    Examples
    --------
    >>> import matplotlib.pyplot as plt  # doctest: +SKIP
    >>> import numpy as np
    >>> from sklearn import metrics
    >>> y = np.array([0, 0, 1, 1])
    >>> pred = np.array([0.1, 0.4, 0.35, 0.8])
    >>> fpr, tpr, thresholds = metrics.roc_curve(y, pred)
    >>> roc_auc = metrics.auc(fpr, tpr)
    >>> display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,\
                                          estimator_name='example estimator')
    >>> display.plot()  # doctest: +SKIP
    >>> plt.show()      # doctest: +SKIP
    """
    # @_deprecate_positional_args
    def __init__(self, *, fpr, tpr,
                 roc_auc=None, estimator_name=None, pos_label=None):
        self.estimator_name = estimator_name
        self.fpr = fpr
        self.tpr = tpr
        self.roc_auc = roc_auc
        self.pos_label = pos_label

    # @_deprecate_positional_args
    def plot(self, ax=None, *, name=None, **kwargs):
        """Plot visualization

        Extra keyword arguments will be passed to matplotlib's ``plot``.

        Parameters
        ----------
        ax : matplotlib axes, default=None
            Axes object to plot on. If `None`, a new figure and axes is
            created.

        name : str, default=None
            Name of ROC Curve for labeling. If `None`, use the name of the
            estimator.

        Returns
        -------
        display : :class:`~sklearn.metrics.plot.RocCurveDisplay`
            Object that stores computed values.
        """

        name = self.estimator_name if name is None else name

        line_kwargs = {}
        if self.roc_auc is not None and name is not None:
            line_kwargs["label"] = f"{name} (AUC = {self.roc_auc:0.4f})"
        elif self.roc_auc is not None:
            line_kwargs["label"] = f"AUC = {self.roc_auc:0.4f}"
        elif name is not None:
            line_kwargs["label"] = name

        line_kwargs.update(**kwargs)

        if ax is None:
            fig, ax = plt.subplots()

        self.line_, = ax.plot(self.fpr, self.tpr, **line_kwargs)
        info_pos_label = (f" (Positive label: {self.pos_label})"
                          if self.pos_label is not None else "")

        xlabel = "False Positive Rate" + info_pos_label
        ylabel = "True Positive Rate" + info_pos_label
        ax.set(xlabel=xlabel, ylabel=ylabel)

        if "label" in line_kwargs:
            ax.legend(loc="lower right")

        self.ax_ = ax
        self.figure_ = ax.figure
        return self