"""Select whole inspection-point bundles without separating paired robot poses."""
import copy
import random
import re


def sample_snapshot_points(snapshot, count=5, seed=None):
    bundles = {}
    for group in snapshot.get('target_groups') or []:
        key = group.get('point_index')
        if key is None:
            match = re.match(r'Point\s+(\d+)\b', group.get('name', ''))
            if not match:
                raise ValueError('포인트를 식별할 수 없는 자세 그룹이 있습니다.')
            key = int(match.group(1)) - 1
        bundles.setdefault(key, []).append(group)
    eligible = {key: groups for key, groups in bundles.items()
                if groups and all(g.get('dda_pose') is not None and g.get('rt_pose') is not None for g in groups)}
    if len(eligible) < count:
        raise ValueError(f'완전한 DDA/RT 자세가 있는 포인트가 {count}개 이상 필요합니다. 현재 {len(eligible)}개입니다.')
    chosen = random.Random(seed).sample(list(eligible), count)
    result = dict(snapshot)
    result['target_groups'] = [copy.deepcopy(g) for key in chosen for g in eligible[key]]
    for i, group in enumerate(result['target_groups']):
        group['index'] = i
    result['random_point_selection'] = {'source_point_indices': chosen, 'seed': seed, 'count': count}
    return result


def sample_snapshot_groups(snapshot, count=5, seed=None, candidate_names=None):
    """Select exactly count complete DDA/RT pose groups, one pair per group."""
    candidates = [g for g in snapshot.get('target_groups') or []
                  if g.get('dda_pose') is not None and g.get('rt_pose') is not None]
    if candidate_names is not None:
        names = set(candidate_names)
        candidates = [g for g in candidates if g.get('name') in names]
        if len(candidates) != len(names):
            raise ValueError('요청한 자세 그룹 중 스냅샷에서 찾을 수 없는 그룹이 있습니다.')
    if len(candidates) < count:
        raise ValueError(f'완전한 DDA/RT 자세 그룹이 {count}개 이상 필요합니다. 현재 {len(candidates)}개입니다.')
    chosen = random.Random(seed).sample(candidates, count)
    result = dict(snapshot)
    result['target_groups'] = [copy.deepcopy(g) for g in chosen]
    for i, group in enumerate(result['target_groups']):
        group['index'] = i
    result['random_group_selection'] = {
        'source_group_names': [g['name'] for g in chosen], 'seed': seed,
        'count': count, 'candidate_count': len(candidates),
    }
    return result
