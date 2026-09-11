"""Training distribution only. Physics, reward and actor sensors are unchanged."""
import numpy as np

FAMILIES = ('shelter','rooms','open','connected-rooms','corridors','multi-exit')
LONG_PLAY = (188, 375, 750)
# Long-play evaluation maps need seeds that no earlier development cohort used
# (the v4 cohort already occupies 1750100000+); a collision must raise, never
# silently drop the maps that exercise the 15/30/60 s regime.
EXTRA_SEED_BASE = 1760100000

def environment_config(rng, index=0, variant="full"):
    # Randomize each reset rather than accidentally assigning the first family
    # whenever only one environment finishes. Evaluation uses disjoint seeds.
    return dict(scenario=str(rng.choice(FAMILIES)), size=float(rng.choice([8,9,10,12])),
        n_boxes=int(rng.integers(2,9)), n_ramps=int(rng.integers(0,3)),
        prep=96, play=144 if variant in ('baseline','short') else int(rng.choice([188,375,750], p=[.3,.4,.3])))

def assign_balanced_roles(rng, count, history_count, seeker_active_fraction=.8):
    # 70% current/current; allocate the remaining fresh episodes to the role
    # that receives fewer actionable samples. Never replay off-policy actions.
    f=float(np.clip(seeker_active_fraction,.1,1))
    hider_only=float(np.clip((f-1+.3)/(1+f),0,.3))
    draw=rng.random(count); old=rng.integers(history_count,size=count)
    roles=np.full((count,2),-1,dtype=np.int64)
    mask=(draw>=.7)&(draw<.7+hider_only);roles[mask,1]=old[mask]
    mask=draw>=.7+hider_only;roles[mask,0]=old[mask]
    return roles

def archive_indices(descriptors, active, limit=24):
    """Keep the fixed anchor, eight newest, diverse behaviors and in-use actors."""
    n=len(descriptors)
    if n<=limit:return list(range(n))
    chosen={*range(min(4,n)),*range(max(0,n-8),n)}
    x=np.asarray(descriptors,dtype=float)
    scale=np.maximum(np.std(x,axis=0),.05);x=x/scale
    while len(chosen)<limit:
        remaining=[i for i in range(n) if i not in chosen]
        distances=np.min(((x[remaining,None]-x[list(chosen)][None])**2).sum(-1),axis=1)
        chosen.add(remaining[int(np.argmax(distances))])
    return sorted(chosen|{int(i) for i in np.asarray(active).ravel() if i>=0})


def extended_cohort(maps, count=24, seed_base=EXTRA_SEED_BASE):
    """Existing fixed maps plus construction layouts with 15/30/60 s play."""
    existing = {m['seed'] for m in maps}
    extra = [dict(seed=seed_base + i, scenario=('connected-rooms', 'corridors', 'multi-exit')[i % 3],
                  arenaConfig=dict(size=8 + i % 5, n_boxes=5 + i % 4, n_ramps=2, play=LONG_PLAY[i % 3]))
             for i in range(count)]
    clash = [m['seed'] for m in extra if m['seed'] in existing]
    if clash:
        raise ValueError(f'Long-play evaluation seeds already exist in the source cohort: {clash[:4]}')
    result = list(maps) + extra
    if len({(m['seed'], m['scenario']) for m in result}) != len(result):
        raise ValueError('Duplicate evaluation map identity')
    return result


def archive_exercised(pilot_updates, snapshot_every, anchors, limit):
    """True when a pilot adds enough snapshots to exceed the archive limit."""
    return anchors + pilot_updates // snapshot_every > limit


def assign_for_variant(rng,count,histories,fraction,variant):
    if variant in ('baseline','unbalanced'):
        from league_ppo import assign_roles
        return assign_roles(rng,count,'B',histories)
    return assign_balanced_roles(rng,count,histories,fraction)
