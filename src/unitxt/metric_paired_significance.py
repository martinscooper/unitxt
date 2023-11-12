import numpy as np
from itertools import combinations
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.contingency_tables import mcnemar
from scipy.stats import permutation_test, ttest_rel, norm
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from collections import namedtuple, defaultdict
import networkx as nx
from copy import deepcopy

COMMON_REPORT_FIELDS = ['model_names', 'metric_name', 'pvalues', 'alternative', 'permute', 'corrected', 'pvalue_is_signif', 'sample_means']

def pval_trans(p):
    # from R emmeans module (emmeans:::.pval.tran)
    # transform a pvalue for visualization
    if isinstance(p, float):
        p = np.array([p])
    bds = (5e-5, 1)
    xx = np.clip(p, a_min=bds[0], a_max=bds[1])
    rtn = norm.cdf(x=np.log(xx), loc=-2.5, scale=3) / 0.7976716
    spec = np.logical_or(p < bds[0], p > bds[1])
    rtn[spec] = p[spec]
    return rtn

def pval_inv(x):
    # from R emmeans module (emmeans:::.pval.inv)
    # transform a value back to the p-value for visualization
    if isinstance(x, float):
        x = np.array([x])
    bds = (5e-5, 0.99995)
    pp = np.clip(x, a_min=bds[0], a_max=bds[1])
    rtn = np.exp(norm.ppf(q=0.7976716 * pp, loc=-2.5, scale=3))
    spec = np.logical_or(x < bds[0], x > bds[1])
    rtn[spec] = x[spec]
    return rtn


class PairedDifferenceTest:
    """
    Class to conduct test of statistical significance (from 0) in average differences between
    sample values for the same observation index.
    Can be used to measure if a pair of prediction models differ substantially in the metric scores they
    receive across observations.
    """

    # output of signif_pair_diff will be a namedtuple in one of these formats
    PERMUTE_REPORT = namedtuple('DiffTest', ' '.join(COMMON_REPORT_FIELDS))
    # additional fields for effect sizes available only if not permute
    TTEST_REPORT = namedtuple('DiffTest', ' '.join(COMMON_REPORT_FIELDS + ['effect_sizes', 'effect_size_is_signif']))

    def __init__(self, nmodels=2, alpha=0.05, model_names=None):
        """
        Args:
            nmodels: the number of models (samples) to be compared pairwise
            alpha: statistical false positive rate confidence to be used in evaluating significance
        """
        self.nmodels = max(2, int(nmodels))
        # desired false positive rate, criterion for pvalues
        self.alpha = float(np.clip(alpha, a_min=1e-10, a_max=0.5))
        self.binary_values = np.array([0,1])
        if model_names is None:
            self.model_names = ['model {}'.format(ii) for ii in range(1, self.nmodels+1)]
        else:
            assert len(model_names) == self.nmodels
            self.model_names = model_names

    @staticmethod
    def _permute_diff_statistic(lx, ly):
        """
        Statistic to be used in permutation test.  Calculates expected difference between pairs (= difference
        in expected values of the samples themselves)
        Args:
            lx: first sample
            ly: second sample
        """
        return np.mean(lx) - np.mean(ly)

    def _check_valid_samples(self, samples_list):
        '''
        Check if samples_list is valid (is of length nmodels), all have same length, and at least 2
        Args:
            samples_list: a list of 1-D numpy arrays
        '''
        # list of equal-length samples greater than size 2
        len_elements = np.unique([len(vec) for vec in samples_list])
        assert (len(samples_list) == self.nmodels) and (len(np.unique(len_elements)) == 1) and (len_elements[0] >= 2)

    def _check_binary(self, samples_list):
        """
        Check if samples are binary-valued
        Args:
            samples_list: a list of 1-D numpy arrays

        Returns:
            boolean
        """
        # return True/False if binary, and returns the binary values
        self._check_valid_samples(samples_list)
        uv = np.unique(np.vstack(samples_list))
        return np.all(np.isin(uv, self.binary_values))

    def _can_use_mcnemar(self, samples_list, alternative='two-sided'):
        """
        Check if can use the McNemar test (accepts only 2 binary equal-length samples) which simplifies computation
        Args:
            samples_list: a list of 1-D numpy arrays
            alternative: alternative hypothesis to be used (only two-sided is valid)

        Returns:
            boolean
        """
        is_binary = self._check_binary(samples_list)
        return is_binary and (len(samples_list) == 2) and (alternative == 'two-sided')

    def _handle_binary_data_contingency(self, samples_list, continuity_correction=True):
        """
        Convert samples into 2x2 contingency table form, even if some value combinations are not observed
        Args:
            samples_list: a list of 1-D numpy arrays
            continuity_correction: whether to perform continuity correction for cells with 0 value

        Returns:
            2x2 numpy cross-tabulation array
        """
        # make sure that a cross tabulation is done that is 2x2 in case some value combinations are missing
        cat_binary = pd.CategoricalDtype(categories=self.binary_values, ordered=False)
        # encode as categorical binary with all values, use dropna=False to avoid dropping missing combinations
        df = pd.DataFrame(np.transpose(np.vstack(samples_list))).astype(cat_binary)
        # need to allow floats for continuity correction
        tbl = pd.crosstab(df[0], df[1], dropna=False).to_numpy()
        return tbl

    def mcnemar_test(self, samples_list):
        """
        Perform McNemar test and return a p-value and effect size
        Use approximation (exact=False) so that effect size can be calculated
        Args:
            samples_list: a list of 1-D numpy arrays (must be binary)

        Returns:
            float
        """
        # assertion in case is used outside of signif_pair_diff
        assert len(samples_list) == 2, "McNemar's test can use only two samples"
        tbl = self._handle_binary_data_contingency(samples_list)
        # Cohen's g effect size, https://rcompanion.org/handbook/H_05.html#_Toc507754342
        # < 0.05 = very small; (0.05, 0.15) = small; (0.15, 0.25) = medium, >= 0.25 is large
        # do continuity correction for the effect size, but not for the p-value
        b, c = max(tbl[0,1], 0.5), max(tbl[1,0], 0.5)
        cohens_g = max(b/(b+c), c/(b+c)) - 0.5
        return mcnemar(table=tbl, exact=True).pvalue, cohens_g

    def iterate_pairs(self, samples_list=None):
        """
        Iterate over ordered combinations of samples or their indices (if samples_list is None)
        Args:
            samples_list: a list of 1-D numpy arrays (must be binary)

        Returns:
            iterator
        """
        # return pairs of values (samples) that are compared, or the indices if samples_list=None
        # notice, the pair order is important if one-sided tests are used because it affects the hypothesis order
        return combinations(range(self.nmodels) if samples_list is None else samples_list, r=2)

    def signif_pair_diff(self, samples_list, metric_name='unknown', alternative='two-sided', permute=False, corrected=True):
        """
        Conduct pairedd-observation est of difference in means between samples
        Args:
            samples_list: a list of 1-D numpy arrays
            metric_name: string of metric name that samples correspond to
            alternative: alternative hypothesis to be used; one-sided (greater/less) should only be used if the arrays have been ordered
            permute: whether or not to use permutation test (requires some simulation)
            corrected: whether or not to apply a correction for control of false positive rate given the number of hypotheses (used when nmodels > 2)

        Returns:
            dict
        """
        # because sample ordering is not fixed, do either greater (one-sided) or two-sided
        assert alternative in ('greater', 'less', 'two-sided')
        if alternative != 'two-sided':
            flds = ('greater', 'descending') if alternative == 'greater' else ('less', 'ascending')
            print("If 'alternative' is '{}, the samples are expected to be sorted in {} order of ANTICIPATED (not OBSERVED) mean value".format(*flds))
        if self.nmodels == 2:
            # require more than 2 hypotheses (only if nmodels > 2) to be able to do a correction
            corrected = False

        # first see if can use McNemar's test, better for binary data; only works for two-sided hypotheses with two samples
        # validity check
        if self._can_use_mcnemar(samples_list, alternative):
            pvalue, eff_size = self.mcnemar_test(samples_list=samples_list)
            permute = False
            res = {'pvalues': np.array([pvalue]), 'pvalue_is_signif': np.array([pvalue <= self.alpha]),
                   'effect_sizes': np.array([eff_size]), 'effect_size_is_signif': np.array([eff_size >= 0.25])}

        else:
            # most other cases if have continuous variables or more than 2 models to compare
            # greater when ttest_rel(a,b) checks if a_i - b_i > 0
            # note, regardless of the significance, the value of the paired expectation E(a_i - b_i) will be the same as the
            # difference in means, i.e. E(a) - E(b).

            # pvalue is low (close to 0) if E(a_i - b_i) is significant (according to altnernative)
            res = {}
            if permute:
                pvalues = np.array([permutation_test(data=vec, statistic=self._permute_diff_statistic, alternative=alternative, permutation_type='samples', vectorized=False).pvalue
                                    for vec in self.iterate_pairs(samples_list=samples_list)])
            else:
                test_res = [ttest_rel(a=vec[0], b=vec[1], alternative=alternative) for vec in self.iterate_pairs(samples_list=samples_list)]
                pvalues = np.array([vv.pvalue for vv in test_res])
                # effect size is the statistic / sqrt(n); equivalent of Cohen's d statistic
                # permutation test don't have effect sizes
                res['effect_sizes'] = np.array([vv.statistic for vv in test_res]) / np.sqrt(np.sqrt(len(samples_list[0])))
                if alternative == 'two-sided':
                    # magnitude in either direction
                    res['effect_size_is_signif'] = np.abs(res['effect_sizes']) >= 0.8
                else:
                    # has to be in the direction specified by the alternative
                    res['effect_size_is_signif'] = res['effect_sizes'] >= 0.8 if alternative == 'greater' else res['effect_sizes'] <= 0.8

            if corrected:
                mult_corr = multipletests(pvals=pvalues, alpha=self.alpha)
                res.update({'pvalues': mult_corr[1], 'pvalue_is_signif': mult_corr[0]})
            else:
                res.update({'pvalues': pvalues, 'pvalue_is_signif': pvalues <= self.alpha})

        res.update({'corrected': corrected, 'alternative': alternative, 'permute': permute, 'sample_means': np.array([np.mean(vec) for vec in samples_list]),
                    'metric_name': metric_name, 'model_names': self.model_names})

        return self.PERMUTE_REPORT(*[res[kk] for kk in self.PERMUTE_REPORT._fields]) if permute else self.TTEST_REPORT(*[res[kk] for kk in self.TTEST_REPORT._fields])

    def _is_valid_signif_report(self, test_res):
        '''
         Args:
            test_res: a namedtuple outputted from signif_pair_diff, from the same item

        Returns:
            boolean
        '''
        assert isinstance(test_res, self.PERMUTE_REPORT) or isinstance(test_res, self.TTEST_REPORT)
        assert len(self.model_names) == len(test_res.model_names)
        assert all([ii == jj for ii, jj in zip(self.model_names, test_res.model_names)]), 'model_names lists must match'

    def lineplot(self, test_res):
        self._is_valid_signif_report(test_res)
        pal = sns.color_palette(palette="Spectral", n_colors=self.nmodels, as_cmap=True)(np.linspace(0, 1, self.nmodels))
        level_order = np.argsort(test_res.sample_means[::-1])
        # pal now accepts the original sample index
        pal = [tuple(pal[ii]) for ii in level_order]

        tick_vals = np.unique(np.array([1e-5, 1e-4, 1e-3, 1e-2, 0.5, 0.1, 0.2, 0.5, 1.0, self.alpha]))
        pval_range = np.array([test_res.pvalues.min(), test_res.pvalues.max()])
        tick_vals = np.array([vv for vv in tick_vals if vv >= pval_range[0] and vv <= pval_range[1]])
        xaxis_range = pval_trans(pval_range)
        obs_range_len = np.diff(xaxis_range)
        range_pad = 0.05

        yaxis_range = np.array([test_res.sample_means.min(), test_res.sample_means.max()])
        yaxis_range = tuple(yaxis_range + range_pad * np.array([-1,1]) * np.diff(yaxis_range))

        xaxis_range = tuple(xaxis_range + range_pad * np.array([-1., 1]) * np.diff(xaxis_range))

        # transform the p-values so that the important (low) ones are separated out more
        pval_transformed = pval_trans(test_res.pvalues)

        dfs = [pd.DataFrame({'pvalue': np.repeat(a=pp, repeats=2),
                             'pvalue_trans': np.repeat(a=ppt, repeats=2),
                             'group': np.array(idxs),
                             'ends': test_res.sample_means[np.array(idxs)], 'color': [pal[ii] for ii in idxs]})#.sort_values(by=['ends'])
                            for idxs, pp, ppt in zip(self.iterate_pairs(), test_res.pvalues, pval_transformed)]
        for ii, df in enumerate(dfs):
            dfs[ii]["midpoint"] = np.repeat(np.max(df['ends']) - 0.5 * np.abs(np.diff(df['ends'])), 2)

        g = sns.scatterplot(data=pd.DataFrame({"mean": test_res.sample_means.mean(), "p-value": [-1, -1]}),
                            x="p-value", y="mean")
        # indicate significance area
        # g.axes.axvspan(axis_range[0], pval_trans(self.alpha), facecolor='lightgray')
        g.axes.axvline(x=pval_trans(self.alpha)[0], ymin=0, ymax=1, linestyle="dashed", color='black')
        # arrow always goes from the first to second
        arrow_sym = '<->' if test_res.alternative == 'two-sided' else '->'
        symbol_dict = {'two-sided': '!=', 'less': '<', 'greater': '>'}
        title = '{} test model A {} model B'.format(test_res.alternative, symbol_dict[test_res.alternative])
        if test_res.alternative != 'two-sided':
            title += '; arrow head goes A -> B'

        for segs in dfs:
            for ii, row in segs.iterrows():
                # g.axes.vlines(x=row['pvalue'], ymin=min(row['ends'], row['midpoint']), ymax=max(row['ends'], row['midpoint']), color=row["color"])
                # if two-sided, each line segment has an arrow starting from the midpoint (xytext) and ending at the endpoint
                # if one-sided, the first row has a segment without an arrow
                arrow_sym = '->' if ((test_res.alternative == 'two-sided') or (ii == 1)) else '-'
                plt.annotate(text='', xytext=(row['pvalue'], row['midpoint']), xy=(row['pvalue'], row['ends']), arrowprops=dict(arrowstyle=arrow_sym, shrinkA=0, shrinkB=0, color=row['color']))
        g.set(xlim=xaxis_range, ylim=yaxis_range, title=title)
        g.axes.set_xticks(ticks=pval_trans(tick_vals), labels=tick_vals)
        g.axes.tick_params(right=True, left=True, labelright=True, labelleft=False)

        # label sample means
        for lev, colpal in zip(level_order, pal):
            g.text(x=xaxis_range[0], y=test_res.sample_means[lev], s=self.model_names[lev], fontdict=dict(color=colpal, horizontalalignment="right"))
        plt.show()

    def _is_valid_signif_report_list(self, test_results_list: list):
        assert isinstance(test_results_list, list)
        # verify same models are being compared, compare to the base
        for vv in test_results_list:
            self._is_valid_signif_report(vv)

        # all have same correction
        assert len(set([vv.corrected for vv in test_results_list])) == 1, "must all have same setting of 'corrected'"
        # all have same alternative
        assert len(set([vv.alternative for vv in test_results_list])) == 1, "msust all have same setting of 'alternative"
        # all different metric_names
        assert len(set([vv.metric_name for vv in test_results_list])) == len(test_results_list), "all metric_name must be different"

    def multiple_metrics_significance_heatmap(self, test_results_list: list):
        """
        Summarize comparisons of multiple models across at least one metric; all metrics must be done on the same model comparisons
        Args:
            test_results_list: list where each element corresponds to a different metric_name result of signif_pair_diff on the same
            set of models compared

        Returns:

        """
        self._is_valid_signif_report_list(test_results_list)
        alternative = test_results_list[0].alternative

        # combine in a matrix

        alt_dict = {'two-sided': {'symbol': 'vs', 'name': 'two-sided'},
                'less': {'symbol': '<', 'name': 'lower-tailed'},
                'greater': {'symbol': '>', 'name': 'upper-tailed'}}
        symb_format = '{} ' + alt_dict[alternative]['symbol'] + ' {}'
        combined_results = pd.DataFrame({vv.metric_name: vv.pvalues for vv in test_results_list},
                                        index=[symb_format.format(a, b) for (a, b) in self.iterate_pairs()])
        combined_mat = combined_results.to_numpy()
        combined_mat[combined_mat > self.alpha] = np.nan
        fig, ax = plt.subplots(1, 1)
        img = ax.matshow(combined_mat, vmin=0, vmax=self.alpha, cmap="Oranges_r", aspect='auto')
        fig.colorbar(img)
        for (j, i), val in np.ndenumerate(combined_mat):
            if not np.isnan(val):
                # use white if background would be too dark
                ax.text(i, j, '{:.3f}'.format(val), ha='center', va='center', fontsize='large',
                        color='white' if val < 0.4 * self.alpha else 'black')
        ax.set_xticks(np.arange(combined_mat.shape[1]))
        ax.set_xticklabels(combined_results.columns)
        ax.set_yticks(np.arange(combined_mat.shape[0]))
        ax.set_yticklabels(combined_results.index)
        ax.set_ylabel('models compared')
        plt.title('{} model comparison significant p-values'.format(alt_dict[alternative]['name']))

        plt.table(cellText=[[vv] for vv in test_results_list[0].model_names],
                  rowLabels=list(range(len(test_results_list[0].model_names))),
                  colLabels=['model name'], loc='bottom', cellLoc='left', colLoc='left', edges='open')
        plt.subplots_adjust()#bottom=0.2)
        plt.show()

    def metric_significant_pairs_graph(self, test_res, use_pvalues=True, model_name_split_char=None):
        '''
        Compare models across a given metric; show a graph where nodes correspond to models, and edgs are drawn
        between pairs that have a significant difference result
        Args:
            test_res: object ouputted from signif_pair_diff
            use_pvalues: boolean, if True use p-values, otherwise effect sizes, to decide if significant

        '''
        self._is_valid_signif_report(test_res)
        # random seed so positions results are the same across runs
        sd = 5
        np.random.seed(sd)
        edges = defaultdict(list)
        criterion = 'pvalue' if use_pvalues else 'effect_size'
        for pair, is_signif in zip(self.iterate_pairs(), getattr(test_res, '{}_is_signif'.format(criterion))):
            if not is_signif:
                # only plot not significant connections
                edges[pair[0]].append(pair[1])

        G = nx.DiGraph()
        for node, others in edges.items():
            if len(others):
                for jj in others:
                    G.add_edge(node, jj)
            else:
                G.add_node(node)

        direction = {'two-sided': 'A vs B', 'less': 'A < B', 'greater': 'A > B'}
        pos = nx.spring_layout(G, seed=sd, k=10 / np.sqrt(G.order()))
        pos = {ii: np.array([pos[ii][0], sm]) for ii, sm in enumerate(test_res.sample_means)}

        model_names = deepcopy(self.model_names)
        if model_name_split_char is not None:
            model_names = [mn.replace(model_name_split_char, "\n") for mn in model_names]
        # rename the keys
        pos = {mn: pos[ii] for ii, mn in enumerate(model_names)}

        G = nx.relabel_nodes(G, mapping={ii: name for ii, name in enumerate(model_names)})
        # retrieve plotted positions, one for each node (model)

        x_coords = [cc[0] for cc in pos.values()]
        x_margin = (max(x_coords) - min(x_coords)) * 0.25

        fig, ax = plt.subplots(1, 1)
        nx.draw_networkx(G, with_labels=True, pos=pos, ax=ax, node_color="lightblue", arrows=test_res.alternative != 'two-sided')
        ax.tick_params(left=True, bottom=False, labelleft=True, labelbottom=False)
        ax.set_ylabel('mean model {}'.format(test_res.metric_name))
        # add some space to x limits so label names don't
        xlims = ax.get_xlim()
        ax.set_xlim(xlims[0] - x_margin, xlims[1] + x_margin)
        ax.set_title(
            'Model {}: edges connect models with insignificant {} {}'.format(test_res.metric_name, direction[test_res.alternative], criterion.replace('_', ' ')))

        plt.show()
