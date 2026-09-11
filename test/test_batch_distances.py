import math
import numpy as np
import pytest
from icir_cleanroom.gas_mapping.mapping.grid_geometry import GridGeometry
from icir_cleanroom.gas_mapping.mapping.path_distance import SamplingDistanceOracle
from icir_cleanroom.gas_mapping.application.hrs import HrsCandidate, HrsManager


def oracle():
    free=np.ones((20,24),dtype=bool)
    free[:16,10]=False
    free[4:12,16]=False
    return SamplingDistanceOracle(GridGeometry(24,20,.05,0.,0.),free)


def test_batch_matches_pairwise_around_obstacles_from_multiple_origins():
    old=oracle();new=oracle()
    points=[new.geometry.cell_center(int(r),int(c)) for r,c in np.argwhere(new.free_mask)[::13]]
    for start in [(1,2),(18,18),(2,20)]:
        xy=new.geometry.cell_center(*start)
        expected=[old.distance(xy,p) for p in points]
        assert new.distances_from(xy,points)==pytest.approx(expected,abs=1e-10)


def test_one_traversal_for_all_targets_and_bounded_origin_cache(monkeypatch):
    o=oracle();compute=o._compute_cell_distances;calls=[]
    def counted(cell):calls.append(cell);return compute(cell)
    monkeypatch.setattr(o,'_compute_cell_distances',counted)
    points=[o.geometry.cell_center(int(r),int(c)) for r,c in np.argwhere(o.free_mask)]
    a=o.geometry.cell_center(2,2);b=o.geometry.cell_center(18,18)
    o.distances_from(a,points);o.distances_from(a,points)
    assert calls==[(2,2)]
    o.distances_from(b,points)
    assert calls==[(2,2),(18,18)]
    assert o._distances=={}  # HRS does not accumulate one full array per target
    assert o._current_origin==(18,18)


def test_same_cell_uses_actual_coordinates_even_when_origin_cell_is_cached():
    o=oracle()
    assert o.distances_from((.11,.11),[(.12,.12)])==(math.dist((.11,.11),(.12,.12)),)
    assert o.distances_from((.13,.13),[(.12,.12)])==(math.dist((.13,.13),(.12,.12)),)


def test_disconnected_corner_and_new_map_invalidation():
    geo=GridGeometry(2,2,.05,0.,0.)
    blocked=SamplingDistanceOracle(geo,np.array([[True,False],[False,True]]))
    with pytest.raises(ValueError,match='no sampling-domain path'):
        blocked.distances_from((.025,.025),[(.075,.075)])
    fresh=SamplingDistanceOracle(geo,np.ones((2,2),bool))
    assert fresh.distances_from((.025,.025),[(.075,.075)])==pytest.approx([math.sqrt(2)*.05])
    assert fresh.distances_from((999,999),[])==()
    with pytest.raises(ValueError,match='outside'):
        fresh.distances_from((999,999),[(.025,.025)])


def test_dd_ucb_scores_and_selected_cell_match_without_pairwise_calls():
    o=oracle();xy=o.geometry.cell_center(18,18)
    candidates=tuple(HrsCandidate(i,0,i,*o.geometry.cell_center(2,i),
                                 score=i/8,mean=i/8,variance=0.,ucb=i/8) for i in range(8))
    expected,target=HrsManager.select_candidate(candidates,xy,o.distance,.03)
    def forbidden(*args):raise AssertionError('pairwise distance must not be called')
    actual,selected=HrsManager.select_candidate(candidates,xy,forbidden,.03,
        distances=o.distances_from(xy,((c.x,c.y) for c in candidates)))
    assert selected.variable==target.variable
    assert [c.score for c in actual]==pytest.approx([c.score for c in expected],abs=1e-10)
