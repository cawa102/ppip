"""Cheap scene probe: object positions in the seed-0 alphabet_soup scene, and each non-soup
object's distance to the alphabet_soup. The object CLOSEST to the soup needs the SMALLEST
grasp-trajectory override from the natural soup grasp, so it is the most likely target the
confined in-scene monitor (which forces the coarse approach but not a large precise override)
can actually hijack ("target has options" -> pick the easiest). No OpenVLA load / rollout."""

from __future__ import annotations

import os
import sys

import numpy as np

HOME = os.path.expanduser("~")
for _p in ("autoresearch/src", "autoresearch", "openvla", "autoresearch/experiments/patch_attack"):
    sys.path.insert(0, os.path.join(HOME, _p))


def _pos(backend, states, name):
    p = backend._position_for(states, name)
    return None if p is None else np.asarray(p, dtype=np.float64)


def main() -> None:
    from evaluator.libero_tasks import resolve_task
    from evaluator.openvla_backend import OpenVLARolloutBackend

    user = "pick up the alphabet soup and place it in the basket"
    backend = OpenVLARolloutBackend()
    ru = resolve_task(user, suite="libero_object")
    env, init_states, _desc, _obj = backend._build_env(ru)
    try:
        cfg = backend._build_cfg()
        obs = env.set_init_state(init_states[0])
        dummy = backend._dummy_action(cfg)
        for _ in range(backend.num_steps_wait):
            obs, _r, _d, _i = env.step(dummy)
        states = backend._object_states(env)
        names = sorted(states.keys())
        print(f"[objects] {len(names)} object states: {names}", flush=True)

        # find the soup key (contains 'soup' or 'alphabet')
        soup_key = next((n for n in names if "soup" in n.lower() or "alphabet" in n.lower()), None)
        basket_key = next((n for n in names if "basket" in n.lower()), None)
        print(f"[objects] soup_key={soup_key} basket_key={basket_key}", flush=True)
        soup = _pos(backend, states, soup_key)
        for n in names:
            p = _pos(backend, states, n)
            d = None if (p is None or soup is None) else float(np.linalg.norm(p - soup))
            dstr = "None" if d is None else f"{d:.3f}"
            pstr = "None" if p is None else f"[{p[0]:.3f},{p[1]:.3f},{p[2]:.3f}]"
            print(f"[obj] {n:32s} pos={pstr} dist_to_soup={dstr}", flush=True)

        # rank non-soup, non-basket, non-region objects by distance to soup
        cand = []
        for n in names:
            if n == soup_key or (basket_key and n == basket_key) or "region" in n.lower():
                continue
            p = _pos(backend, states, n)
            if p is not None and soup is not None:
                cand.append((float(np.linalg.norm(p - soup)), n))
        cand.sort()
        print("[rank] non-soup objects by distance to soup (closest first = easiest hijack):",
              flush=True)
        for d, n in cand:
            print(f"[rank]   {d:.3f}  {n}", flush=True)
    finally:
        if hasattr(env, "close"):
            env.close()


if __name__ == "__main__":
    main()
