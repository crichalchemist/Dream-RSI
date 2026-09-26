"""Reading the replay pool (trace_pool/iter*/) with prefix-safe planning contexts."""

import glob
import json
import os

from see.policy.api import GridPlanningContext
from see.world import Trace


def load_pool(pool_dir: str) -> list:
    """Return [(trace, manifest_or_None)] ordered by live iteration."""
    out = []
    for d in sorted(glob.glob(os.path.join(pool_dir, "iter*"))):
        trace = Trace.load(os.path.join(d, "trace.json"))
        mpath = os.path.join(d, "live_cycle_manifest.json")
        manifest = None
        if os.path.exists(mpath):
            with open(mpath) as f:
                manifest = json.load(f)
        out.append((trace, manifest))
    return out


def context_factory(pool: list, fallback: tuple, hard_max: tuple):
    """Replaying world i, plan_grid sees only the manifests of cycles before i."""
    index = {t.trace_id: i for i, (t, _) in enumerate(pool)}
    manifests = [m for _, m in pool]

    def context_for(trace: Trace) -> GridPlanningContext:
        earlier = tuple(m for m in manifests[: index[trace.trace_id]] if m is not None)
        tb, tr = trace.grid
        return GridPlanningContext(
            earlier,
            fallback[0],
            fallback[1],
            hard_max[0],
            hard_max[1],
            trace.max_parallelism,
            trace_branch_count=tb,
            trace_refine_count=tr,
        )

    return context_for


def next_context(
    pool: list, fallback: tuple, hard_max: tuple, max_parallelism: int
) -> GridPlanningContext:
    """The context online() plans the next live cycle with: every manifest, no trace fields."""
    history = tuple(m for _, m in pool if m is not None)
    return GridPlanningContext(
        history, fallback[0], fallback[1], hard_max[0], hard_max[1], max_parallelism
    )
