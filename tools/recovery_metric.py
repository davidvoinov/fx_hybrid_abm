"""
Resiliency measurement for the spread series, following Large (2007).

Large measures resiliency in two parts, namely the probability that the book
replenishes at all and, conditional on replenishment, the speed at which it
does so. We reproduce that structure here and normalise the excess spread by
its realised peak, so that the speed component cannot mechanically favour the
arm whose peak is lower. Lo and Hall (2015) report the same statistic, a
half-life of spread normalisation, which is also the statistic the project's
calibration target is written against.

Inputs
    output/resilience/final_300seed_raw.npz
        base_traj  (n_seeds, n_ticks)  spread path, dealer only arm
        comp_traj  (n_seeds, n_ticks)  spread path, arm with the facility

Outputs
    output/resilience/recovery_metric.json
    output/resilience/recovery_metric.txt
    output/resilience/recovery_normalised_irf.png
"""
from __future__ import annotations
import json, os
import numpy as np

NPZ = 'output/resilience/final_300seed_raw.npz'
OUTDIR = 'output/resilience'
FRAC = 0.50           # half-life
PRE_PAD = 5           # ticks dropped just before the shock
SMOOTH = 5            # moving average, ticks
BOOT = 4000
SEED = 42


def _smooth(a: np.ndarray, w: int = SMOOTH) -> np.ndarray:
    if w <= 1:
        return a
    k = np.ones(w) / w
    return np.apply_along_axis(lambda r: np.convolve(r, k, mode='same'), 1, a)


def detect_shock(base: np.ndarray, comp: np.ndarray) -> int:
    """Shock index inside the stored window, taken as the first tick where the
    cross seed median rises decisively above its own pre shock level."""
    med = np.nanmedian(np.vstack([base, comp]), axis=0)
    head = med[:30]
    thr = np.nanmedian(head) + 4.0 * np.nanstd(head)
    above = np.where(med > thr)[0]
    return int(above[0]) - 1 if len(above) else 30


def per_seed(tr: np.ndarray, shock: int, frac: float = FRAC):
    """Peak excess, recovery indicator and normalised half-life per seed."""
    tr = _smooth(tr)
    pre = np.nanmedian(tr[:, :max(shock - PRE_PAD, 5)], axis=1)
    peaks, halves, recovered = [], [], []
    for i in range(tr.shape[0]):
        seg = tr[i, shock:] - pre[i]
        if not np.isfinite(seg).any():
            continue
        pk = int(np.nanargmax(seg))
        peak = seg[pk]
        if peak <= 0.5:            # no material dislocation on this seed
            continue
        peaks.append(peak)
        tail = seg[pk:]
        hit = np.where(tail < frac * peak)[0]
        if len(hit):
            recovered.append(1)
            halves.append(int(hit[0]))
        else:
            recovered.append(0)
            halves.append(np.nan)
    return np.array(peaks), np.array(recovered), np.array(halves, dtype=float)


def normalised_irf(tr: np.ndarray, shock: int, horizon: int = 150):
    """Excess spread normalised by its own peak and aligned on that peak.

    Each seed is shifted so that tick zero is its own maximum, which keeps the
    median curve from understating the peak when seeds peak at different times.
    Returns the median curve and the per seed normalised cumulative excess,
    the latter being a continuous measure of persistence relative to size.
    """
    tr = _smooth(tr)
    pre = np.nanmedian(tr[:, :max(shock - PRE_PAD, 5)], axis=1)
    rows, areas = [], []
    for i in range(tr.shape[0]):
        seg = tr[i, shock:] - pre[i]
        if not np.isfinite(seg).any():
            continue
        pk_i = int(np.nanargmax(seg))
        pk = seg[pk_i]
        if pk <= 0.5:
            continue
        tail = seg[pk_i:pk_i + horizon] / pk
        if len(tail) < horizon:
            tail = np.pad(tail, (0, horizon - len(tail)), constant_values=np.nan)
        rows.append(tail)
        areas.append(np.nansum(np.clip(tail, 0.0, None)))
    return np.nanmedian(np.vstack(rows), axis=0), np.array(areas)


def paired_boot(a: np.ndarray, b: np.ndarray, rng, n: int = BOOT):
    """Bootstrap CI for the paired median difference b minus a."""
    ok = np.isfinite(a) & np.isfinite(b)
    d = b[ok] - a[ok]
    if len(d) < 5:
        return float('nan'), (float('nan'), float('nan')), 0
    draws = [np.median(d[rng.integers(0, len(d), len(d))]) for _ in range(n)]
    return float(np.median(d)), (float(np.percentile(draws, 2.5)),
                                 float(np.percentile(draws, 97.5))), int(len(d))


def main():
    d = np.load(NPZ, allow_pickle=True)
    base, comp = d['base_traj'], d['comp_traj']
    shock = detect_shock(base, comp)
    rng = np.random.default_rng(SEED)

    pb, rb, hb = per_seed(base, shock)
    pc, rc, hc = per_seed(comp, shock)
    n = min(len(hb), len(hc))

    dh, ci_h, n_h = paired_boot(hb[:n], hc[:n], rng)
    dp, ci_p, _ = paired_boot(pb[:n], pc[:n], rng)
    irf_b, area_b = normalised_irf(base, shock)
    irf_c, area_c = normalised_irf(comp, shock)
    na = min(len(area_b), len(area_c))
    da, ci_a, _ = paired_boot(area_b[:na], area_c[:na], rng)

    res = {
        'shock_index_in_window': shock,
        'n_seeds': int(n),
        'half_life_definition': f'ticks from peak until normalised excess falls below {FRAC}',
        'amplitude': {
            'peak_excess_bps_without': float(np.median(pb)),
            'peak_excess_bps_with': float(np.median(pc)),
            'paired_median_change': dp, 'ci95': ci_p,
        },
        'recovery_probability': {
            'without': float(np.mean(rb)), 'with': float(np.mean(rc)),
        },
        'normalised_cumulative_excess': {
            'definition': 'area under the peak aligned, peak normalised excess over 150 ticks',
            'without': float(np.median(area_b)), 'with': float(np.median(area_c)),
            'paired_median_change': da, 'ci95': ci_a,
        },
        'normalised_half_life_ticks': {
            'without': float(np.nanmedian(hb)), 'with': float(np.nanmedian(hc)),
            'paired_median_change': dh, 'ci95': ci_h, 'n_paired': n_h,
            'share_faster_with_facility': float(np.nanmean(
                (hc[:n] - hb[:n]) < 0)),
        },
    }
    os.makedirs(OUTDIR, exist_ok=True)
    with open(f'{OUTDIR}/recovery_metric.json', 'w') as f:
        json.dump(res, f, indent=2)

    lines = [
        'Resiliency of the quoted spread, two part measure following Large (2007)',
        f'  shock index in stored window      {shock}',
        f'  seeds with a material dislocation {n}',
        '',
        'Amplitude, peak excess over the pre shock level, bps',
        f'  without facility                  {res["amplitude"]["peak_excess_bps_without"]:.2f}',
        f'  with facility                     {res["amplitude"]["peak_excess_bps_with"]:.2f}',
        f'  paired median change              {dp:+.2f}  95% CI [{ci_p[0]:+.2f}, {ci_p[1]:+.2f}]',
        '',
        'Probability of recovery within the stored window',
        f'  without facility                  {res["recovery_probability"]["without"]:.3f}',
        f'  with facility                     {res["recovery_probability"]["with"]:.3f}',
        '',
        f'Normalised half life, ticks, conditional on recovery',
        f'  without facility                  {res["normalised_half_life_ticks"]["without"]:.2f}',
        f'  with facility                     {res["normalised_half_life_ticks"]["with"]:.2f}',
        f'  paired median change              {dh:+.2f}  95% CI [{ci_h[0]:+.2f}, {ci_h[1]:+.2f}]',
        f'  share of seeds faster with        {res["normalised_half_life_ticks"]["share_faster_with_facility"]:.3f}',
        '',
        'Normalised cumulative excess, area under the peak aligned response',
        f'  without facility                  {res["normalised_cumulative_excess"]["without"]:.2f}',
        f'  with facility                     {res["normalised_cumulative_excess"]["with"]:.2f}',
        f'  paired median change              {da:+.2f}  95% CI [{ci_a[0]:+.2f}, {ci_a[1]:+.2f}]',
    ]
    txt = '\n'.join(lines)
    with open(f'{OUTDIR}/recovery_metric.txt', 'w') as f:
        f.write(txt + '\n')
    print(txt)

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(irf_b, label='without facility', lw=1.8)
        ax.plot(irf_c, label='with facility', lw=1.8)
        ax.axhline(FRAC, ls='--', lw=0.9, color='grey')
        ax.set_xlabel('ticks since own peak')
        ax.set_ylabel('excess spread, normalised by own peak')
        ax.set_title('Normalised impulse response of the quoted spread')
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(f'{OUTDIR}/recovery_normalised_irf.png', dpi=160)
        print(f'\nfigure written to {OUTDIR}/recovery_normalised_irf.png')
    except Exception as e:
        print('figure skipped:', e)


if __name__ == '__main__':
    main()
