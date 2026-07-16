"""Contract of the phase-aware target-progress metric (Task 3, CPU-pure).

Bare `min_target_dist` (target->basket distance) is blunt: it does not move while the
arm is still *approaching* a stationary target, so it cannot tell "making progress" from
"stuck". The phased signal reports which stage of the pick-place the attack has reached
(approach -> grasp -> carry/displacement -> containment) plus a within-phase scalar, so
the Task-6 inner outcome gate can see approach progress the blunt metric misses.

All hand-built states -- no OpenVLA, no simulator.
"""

from __future__ import annotations

_TGT = "salad_dressing_1"
_BKT = "basket_region"


def _states(target_pos, basket_pos):
    return {_TGT: {"pos": tuple(target_pos)}, _BKT: {"pos": tuple(basket_pos)}}


class _FakeObjectState:
    """Mimics LIBERO's ObjectState: xyz is read via get_geom_state()['pos'], not indexing."""

    def __init__(self, pos):
        self._pos = tuple(pos)

    def get_geom_state(self):
        return {"pos": self._pos, "quat": (1.0, 0.0, 0.0, 0.0)}


def test_phase_progress_reads_a_libero_objectstate_via_get_geom_state():
    from progress_metrics import Phase, phase_progress

    # The live env's object_states_dict holds ObjectState objects (not dicts/arrays); the
    # metric must read their position without crashing (regression: run_oracle GPU smoke).
    states = {
        _TGT: _FakeObjectState((0.0, 0.0, 0.90)),
        _BKT: _FakeObjectState((0.30, 0.30, 0.90)),
    }

    result = phase_progress(
        states, eef_pose=(0.0, 0.0, 1.20), target_obj=_TGT,
        basket_region=_BKT, initial_target_pos=(0.0, 0.0, 0.90),
    )

    # Target stationary, eef far above -> APPROACH, scalar = eef->target distance (0.30).
    assert result.phase is Phase.APPROACH
    assert abs(result.scalar - 0.30) < 1e-9


def test_approach_phase_scalar_is_eef_to_target_and_improves_as_eef_nears():
    from progress_metrics import Phase, phase_progress

    target_pos = (0.0, 0.0, 0.90)
    basket_pos = (0.30, 0.30, 0.90)
    states = _states(target_pos, basket_pos)

    far = phase_progress(
        states, eef_pose=(0.0, 0.0, 1.30), target_obj=_TGT,
        basket_region=_BKT, initial_target_pos=target_pos,
    )
    near = phase_progress(
        states, eef_pose=(0.0, 0.0, 1.00), target_obj=_TGT,
        basket_region=_BKT, initial_target_pos=target_pos,
    )

    assert far.phase is Phase.APPROACH
    assert near.phase is Phase.APPROACH
    # Within approach the scalar is the eef->target distance: it improves (shrinks) as
    # the arm nears, even though the target has not moved (the blunt metric stays put).
    assert near.scalar < far.scalar


def test_grasp_phase_when_eef_reaches_a_still_stationary_target():
    from progress_metrics import Phase, phase_progress

    target_pos = (0.0, 0.0, 0.90)
    states = _states(target_pos, (0.30, 0.30, 0.90))

    grasped = phase_progress(
        states, eef_pose=(0.005, 0.0, 0.905), target_obj=_TGT,
        basket_region=_BKT, initial_target_pos=target_pos,
    )

    # eef is essentially on the target but the target has not moved yet -> GRASP, not CARRY.
    assert grasped.phase is Phase.GRASP


def test_carry_phase_when_target_is_displaced_en_route_to_basket():
    from progress_metrics import Phase, phase_progress

    initial = (0.0, 0.0, 0.90)
    basket = (0.30, 0.30, 0.90)
    carried = (0.15, 0.15, 1.00)  # lifted + moved halfway to the basket
    states = _states(carried, basket)

    result = phase_progress(
        states, eef_pose=carried, target_obj=_TGT,
        basket_region=_BKT, initial_target_pos=initial,
    )

    assert result.phase is Phase.CARRY
    # Now the informative scalar is target->basket distance (progress toward the goal).
    assert abs(result.scalar - ((0.15**2 + 0.15**2 + 0.10**2) ** 0.5)) < 1e-9


def test_containment_phase_when_displaced_target_is_inside_the_basket_region():
    from progress_metrics import Phase, phase_progress

    initial = (0.0, 0.0, 0.90)
    basket = (0.30, 0.30, 0.90)
    placed = (0.30, 0.30, 0.91)  # displaced from rest AND essentially at the basket
    states = _states(placed, basket)

    result = phase_progress(
        states, eef_pose=placed, target_obj=_TGT,
        basket_region=_BKT, initial_target_pos=initial,
    )

    assert result.phase is Phase.CONTAINMENT


def test_containment_requires_displacement_not_just_region_membership():
    from progress_metrics import Phase, phase_progress

    # Target sits at the basket but has NOT been displaced from where it started
    # (initial == current): this must NOT count as a placement (guards a false hijack).
    basket = (0.30, 0.30, 0.90)
    at_basket = (0.30, 0.30, 0.90)
    states = _states(at_basket, basket)

    result = phase_progress(
        states, eef_pose=(0.0, 0.0, 1.30), target_obj=_TGT,
        basket_region=_BKT, initial_target_pos=at_basket,
    )

    assert result.phase is not Phase.CONTAINMENT


def test_phases_advance_monotonically_along_a_pick_place_trajectory():
    from progress_metrics import phase_progress

    initial = (0.0, 0.0, 0.90)
    basket = (0.30, 0.30, 0.90)
    # (target_pos, eef_pos) snapshots along approach -> grasp -> carry -> place.
    trajectory = [
        ((0.0, 0.0, 0.90), (0.0, 0.0, 1.30)),   # approaching
        ((0.0, 0.0, 0.90), (0.01, 0.0, 0.91)),  # at target
        ((0.15, 0.15, 1.00), (0.15, 0.15, 1.00)),  # carrying
        ((0.30, 0.30, 0.91), (0.30, 0.30, 0.91)),  # placed
    ]
    phases = [
        phase_progress(
            _states(t, basket), eef_pose=e, target_obj=_TGT,
            basket_region=_BKT, initial_target_pos=initial,
        ).phase
        for t, e in trajectory
    ]

    # Phase is ordered (IntEnum): the trajectory never regresses to an earlier phase.
    assert [int(p) for p in phases] == sorted(int(p) for p in phases)
    assert int(phases[0]) < int(phases[-1])
