"""Extrema anchors followed by endpoint-constrained, full-coverage segment DTW.

All joints share one time mapping. Per-joint extrema that contradict that ordering
are reported, not silently forced into a physically inconsistent 6D pose.
"""
import numpy as np
from scipy.signal import find_peaks
from scipy.spatial.distance import cdist


def extrema(q, prominence):
    events = []
    for joint in range(q.shape[1]):
        for kind, sign in (('max', 1), ('min', -1)):
            indices, props = find_peaks(sign*q[:, joint], prominence=prominence, plateau_size=True)
            for k, index in enumerate(indices):
                events.append(dict(index=int(index), joint=joint, kind=kind,
                                   prominence=float(props['prominences'][k])))
    return sorted(events, key=lambda e: (e['index'], e['joint'], e['kind']))


def _event_pairs(ref, meas, rt, mt, max_shift):
    """Order-preserving event alignment, same joint/type, with explicit gaps.

    Match by event order and normalized time, not by equality of peak heights:
    matching peaks by amplitude would hide the very deficit being measured.
    """
    nr, nm = len(ref), len(meas)
    cost = np.full((nr+1, nm+1), np.inf)
    cost[:, 0] = np.arange(nr+1)*.6
    cost[0, :] = np.arange(nm+1)*.6
    choices = np.zeros((nr+1, nm+1), dtype=np.uint8)
    for i in range(1, nr+1):
        for j in range(1, nm+1):
            a, b = ref[i-1], meas[j-1]
            shift = abs(rt[a['index']] - mt[b['index']])
            match = shift/max_shift if a['kind'] == b['kind'] and shift <= max_shift else np.inf
            options = (cost[i-1,j-1]+match, cost[i-1,j]+.6, cost[i,j-1]+.6)
            choices[i,j] = int(np.argmin(options))
            cost[i,j] = min(options)
    pairs, missing, extra = [], [], []
    i, j = nr, nm
    while i or j:
        choice = int(choices[i,j]) if i and j else (1 if i else 2)
        if choice == 0:
            pairs.append((ref[i-1],meas[j-1])); i-=1; j-=1
        elif choice == 1:
            missing.append(ref[i-1]); i-=1
        else:
            extra.append(meas[j-1]); j-=1
    return pairs[::-1], missing[::-1], extra[::-1]


def _anchor_chain(records):
    """Maximum-support chain with strictly increasing measured/reference indices."""
    candidates = {}
    for row in records:
        if row['status'] == 'candidate':
            candidates.setdefault((row['meas_index'], row['ref_index']), []).append(row)
    keys = sorted(candidates)
    score, parent = [], []
    for i, (m, r) in enumerate(keys):
        allowed = [j for j in range(i) if keys[j][0] < m and keys[j][1] < r]
        previous = max(allowed, key=lambda j: score[j]) if allowed else -1
        score.append(len(candidates[(m,r)]) + (score[previous] if previous >= 0 else 0))
        parent.append(previous)
    chosen = set()
    i = int(np.argmax(score)) if score else -1
    while i >= 0:
        chosen.add(keys[i]); i = parent[i]
    for key, rows in candidates.items():
        for row in rows:
            row['status'] = 'anchor' if key in chosen else 'order_conflict'
    return sorted(chosen)


def _dtw_path(meas, ref):
    """Standard DTW steps: diagonal, measured-only, reference-only; both ends fixed."""
    c = cdist(meas, ref)
    n, m = c.shape
    d = np.empty_like(c)
    d[0] = np.cumsum(c[0])
    # Prefix minimum implements horizontal transitions without a Python inner loop.
    for i in range(1, n):
        prefix = np.cumsum(c[i])
        enter = np.minimum(d[i-1], np.r_[np.inf, d[i-1,:-1]])
        d[i] = prefix + np.minimum.accumulate(enter - (prefix-c[i]))
    i,j = n-1,m-1
    path = [(i,j)]
    while i or j:
        if i == 0:
            j-=1
        elif j == 0:
            i-=1
        else:
            choice = int(np.argmin((d[i-1,j-1],d[i-1,j],d[i,j-1])))
            if choice != 2: i-=1
            if choice != 1: j-=1
        path.append((i,j))
    return path[::-1]


def extrema_dtw(meas, ref, meas_times=None, ref_times=None, prominence=.05, max_shift=.35):
    meas, ref = np.asarray(meas, dtype=float), np.asarray(ref, dtype=float)
    if (meas.ndim != 2 or ref.ndim != 2 or meas.shape[1] != ref.shape[1]
            or len(meas)<2 or len(ref)<2 or not np.all(np.isfinite(meas))
            or not np.all(np.isfinite(ref)) or not np.isfinite(prominence) or prominence < 0
            or not np.isfinite(max_shift) or not 0 < max_shift <= 1):
        raise ValueError('Invalid trajectory or extrema matching settings')

    def times(values, length):
        t = np.arange(length, dtype=float) if values is None else np.asarray(values,dtype=float)
        if t.shape != (length,) or not np.all(np.isfinite(t)) or np.any(np.diff(t)<=0):
            raise ValueError('Matching times must be finite and strictly increasing')
        return t, (t-t[0])/(t[-1]-t[0])

    mt, mn = times(meas_times,len(meas)); rt,rn = times(ref_times,len(ref))
    re,me = extrema(ref,prominence),extrema(meas,prominence)
    records=[]
    for joint in range(ref.shape[1]):
        pairs, missing, extra = _event_pairs([e for e in re if e['joint']==joint],
                                             [e for e in me if e['joint']==joint],rn,mn,max_shift)
        for a,b in pairs:
            r,m = a['index'],b['index']
            signed = float(meas[m,joint]-ref[r,joint])
            records.append(dict(joint=joint+1,kind=a['kind'],ref_index=r,meas_index=m,
                ref_time_s=float(rt[r]-rt[0]),meas_time_s=float(mt[m]-mt[0]),
                ref_deg=float(ref[r,joint]),meas_deg=float(meas[m,joint]),error_deg=signed,
                deficit_deg=-signed if a['kind']=='max' else signed,status='candidate'))
        for source,events in (('reference_missing',missing),('measurement_extra',extra)):
            for e in events:
                is_ref=source=='reference_missing';i=e['index']
                records.append(dict(joint=joint+1,kind=e['kind'],
                    ref_index=i if is_ref else None,meas_index=None if is_ref else i,
                    ref_time_s=float(rt[i]-rt[0]) if is_ref else None,
                    meas_time_s=None if is_ref else float(mt[i]-mt[0]),
                    ref_deg=float(ref[i,joint]) if is_ref else None,
                    meas_deg=None if is_ref else float(meas[i,joint]),
                    error_deg=None,deficit_deg=None,status=source))
    anchors=[(0,0)]+_anchor_chain(records)+[(len(meas)-1,len(ref)-1)]
    index=np.zeros(len(meas),dtype=int)
    full_path=[]
    for (ma,ra),(mb,rb) in zip(anchors[:-1],anchors[1:]):
        path=_dtw_path(meas[ma:mb+1],ref[ra:rb+1])
        # One reference point per measured sample for existing plot/export APIs.
        # Full many-to-many correspondence remains available in dtw_pairs.
        grouped = [[] for _ in range(mb-ma+1)]
        for m,r in path:
            grouped[m].append(r)
        for m,refs in enumerate(grouped):
            best=min(refs,key=lambda r: np.linalg.norm(meas[ma+m]-ref[ra+r]))
            index[ma+m]=ra+best
        full_path.extend((ma+m,ra+r) for m,r in (path if not full_path else path[1:]))
    for m,r in anchors: index[m]=r
    return dict(indices=index,anchors=anchors,extrema=records,dtw_pairs=full_path,
                method='extrema_anchored_dtw_v1',prominence_deg=float(prominence),max_shift=float(max_shift))
