"""Reproduce the saved W=72 maximum-Weibull calibration fits for the thesis."""
from pathlib import Path
import json
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import weibull_min

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.detection.bcad import rolling_absolute_error_max

RUN = 'all6_multi3_seq12_h64_l1_bs64_lr1e-03_wd0_do0_e60_p8_seed42'
SOURCE = ROOT / 'results/kelmarsh/experiments' / RUN
STEM = 'foldedmle_foldedmle_samecenterabs_w72_q990_vm5_c6_calib2021_2022'
OUT = ROOT.parent / 'Thesis_LaTeX/figures/chapter5/bcad_maximum_weibull_calibration_fits'


def select_maxima(rolling, episodes):
    """Match the selection used in main_fit_bcad_weibull_from_folded_mle.py."""
    episodes = episodes[episodes['in_fault_horizon'].fillna(False).astype(bool)].copy()
    for key in ['start_time', 'end_time']:
        episodes[key] = pd.to_datetime(episodes[key])
    selected = []
    for turbine, part in rolling.groupby('turbine_id', sort=False):
        mask = pd.Series(False, index=part.index)
        for ep in episodes[episodes.turbine_id.eq(turbine)].itertuples(index=False):
            mask |= part['Date and time'].between(ep.start_time, ep.end_time, inclusive='both')
        selected.append(part.loc[mask, 'window_max'])
    return pd.concat(selected, ignore_index=True).dropna()


def main():
    params = pd.read_csv(SOURCE / f'bcad_weibull_mle_parameters_from_{STEM}.csv')
    params = params.loc[params.target.eq('Rear bearing temperature (°C)')]
    targets = params.target.tolist()
    cols = ['turbine_id', 'segment_id', 'Date and time'] + [f'error::{t}' for t in targets]
    healthy = pd.read_csv(SOURCE / 'healthy_val_predictions.csv', usecols=cols)
    calibration = pd.read_csv(SOURCE / 'bcad_calibration_2021-05-26_2022_predictions.csv', usecols=cols)
    episodes = pd.read_csv(SOURCE / f'bcad_calibration_ta_episodes_{STEM}.csv')
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.spines.top': True, 'axes.spines.right': True,
                         'pdf.fonttype': 42, 'ps.fonttype': 42})
    # Match the thesis Chapter 6 distribution figures, not presentation styling.
    fig, ax = plt.subplots(figsize=(8.6, 3.8), layout='constrained')
    colors = ['#087f72', '#8053a4']
    fills = ['#d7e2e9', '#edb2ae']
    audit = []
    for record in params.itertuples(index=False):
        h = rolling_absolute_error_max(healthy, record.target, 72)['window_max']
        a = select_maxima(rolling_absolute_error_max(calibration, record.target, 72), episodes)
        assert len(h) == record.healthy_samples and len(a) == record.abnormal_samples
        assert np.isclose(h.median(), record.healthy_median)
        assert np.isclose(a.median(), record.abnormal_median)
        xmax = float(pd.concat([h, a], ignore_index=True).quantile(.995))
        bins = np.linspace(0, xmax, 101)
        for col, (sample, shape, scale, label) in enumerate([
            (h, record.healthy_weibull_shape, record.healthy_weibull_scale, 'H'),
            (a, record.weibull_shape, record.weibull_scale, 'A')]):
            # Bin counts use all samples in their normalisation, even outside the visible range.
            counts, _ = np.histogram(sample, bins=bins)
            ax.stairs(counts / (len(sample) * np.diff(bins)), bins, fill=True,
                      color=fills[col], edgecolor='#95a8b3' if col == 0 else '#c88e89',
                      alpha=.55, linewidth=.45, label=f'{label} window maxima', zorder=1)
            x = np.linspace(max(xmax / 2500, 1e-6), xmax, 1200)
            ax.plot(x, weibull_min.pdf(x, shape, loc=0, scale=scale),
                    color=colors[col], lw=2, label=rf'$f_{label}$', zorder=3)
            ax.set_xlim(0, xmax)
            ax.set_ylim(bottom=0)
            ax.grid(axis='y', alpha=.14)
        audit.append({'target': record.target, 'healthy_samples': len(h), 'alternative_samples': len(a),
                      'healthy_shape': record.healthy_weibull_shape, 'healthy_scale': record.healthy_weibull_scale,
                      'alternative_shape': record.weibull_shape, 'alternative_scale': record.weibull_scale})
    ax.set_title('Rear bearing', fontsize=13)
    ax.set_xlabel('Window-maximum absolute residual (°C)', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.tick_params(labelsize=10)
    ax.legend(frameon=False, fontsize=10, loc='upper right')
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT.with_suffix('.pdf'), bbox_inches='tight')
    fig.savefig(OUT.with_suffix('.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    OUT.with_suffix('.json').write_text(json.dumps({
        'window_size': 72, 'source_parameters': str(SOURCE / f'bcad_weibull_mle_parameters_from_{STEM}.csv'),
        'source_episodes': str(SOURCE / f'bcad_calibration_ta_episodes_{STEM}.csv'),
        'note': 'Saved calibration fits reproduced without refitting; no test-period data used. '
                'Shared displayed limit: pooled H/A 99.5th percentile. Histogram normalisation uses all samples.',
        'targets': audit}, indent=2), encoding='utf-8')
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
