import numpy as np
from statsmodels.stats.power import NormalIndPower, TTestPower
from statsmodels.stats.proportion import proportion_effectsize
from scipy.stats import mcnemar, ttest_rel, wilcoxon

class PowerAnalyzer:
    @staticmethod
    def calculate_proportion_sample_size(p1, p2, alpha=0.05, power=0.8, ratio=1):
        """
        Calculate sample size for two-sample proportion test using Cohen's h.
        p1: Baseline proportion (e.g., 0.7 for 70%)
        p2: Improved proportion (e.g., 0.75 for 75%)
        alpha: Significance level (default 0.05)
        power: Desired power (default 0.8)
        ratio: Ratio of sample sizes (n2 / n1, default 1)
        Returns sample size per group.
        """
        effect_size = proportion_effectsize(p1, p2)
        power_analysis = NormalIndPower()
        sample_size = power_analysis.solve_power(
            effect_size=effect_size,
            alpha=alpha,
            power=power,
            ratio=ratio,
            alternative='two-sided'
        )
        return int(np.ceil(sample_size))

    @staticmethod
    def calculate_continuous_sample_size(effect_size, alpha=0.05, power=0.8, ratio=1):
        """
        Calculate sample size for two-sample t-test.
        effect_size: Cohen's d (difference / pooled std dev)
        alpha: Significance level (default 0.05)
        power: Desired power (default 0.8)
        ratio: Ratio of sample sizes (n2 / n1, default 1)
        Returns sample size per group.
        """
        power_analysis = TTestPower()
        sample_size = power_analysis.solve_power(
            effect_size=effect_size,
            alpha=alpha,
            power=power,
            ratio=ratio,
            alternative='two-sided'
        )
        return int(np.ceil(sample_size))

    @staticmethod
    def test_proportion_significance(baseline, improved, alpha=0.05):
        """
        Perform McNemar's test for paired nominal data.
        baseline: List of 0/1 for baseline model (1=correct)
        improved: List of 0/1 for improved model (1=correct)
        alpha: Significance level (default 0.05)
        Returns p-value and whether result is significant.
        """
        # Build contingency matrix
        a = sum(1 for b, i in zip(baseline, improved) if (b == 1 and i == 1))
        b_ = sum(1 for b, i in zip(baseline, improved) if (b == 1 and i == 0))
        c_ = sum(1 for b, i in zip(baseline, improved) if (b == 0 and i == 1))
        d_ = sum(1 for b, i in zip(baseline, improved) if (b == 0 and i == 0))
        contingency_table = [[a, b_], [c_, d_]]
        result = mcnemar(contingency_table, exact=False)
        p_value = result.pvalue
        significant = p_value < alpha
        return p_value, significant

    @staticmethod
    def test_continuous_significance(baseline, improved, alpha=0.05, test_type='t-test'):
        """
        Perform paired test for continuous data.
        baseline: List of baseline model's scores
        improved: List of improved model's scores
        alpha: Significance level (default 0.05)
        test_type: 't-test' (parametric) or 'wilcoxon' (non-parametric, default 't-test')
        Returns p-value and whether result is significant.
        """
        if test_type == 't-test':
            stat, p_value = ttest_rel(baseline, improved)
        elif test_type == 'wilcoxon':
            stat, p_value = wilcoxon(baseline, improved)
        else:
            raise ValueError("test_type must be 't-test' or 'wilcoxon'")
        significant = p_value < alpha
        return p_value, significant