import pathlib
import sys

import pytest

sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))

from benchmark_path_planners import _parse_method, _safe_name, _split_methods


def test_split_keeps_commas_inside_brackets():
    assert _split_methods("direct_path,direct_path+stomp[w_obs=0,k_att=0],rrt_connect") == [
        "direct_path", "direct_path+stomp[w_obs=0,k_att=0]", "rrt_connect"]


def test_split_ignores_blank_entries():
    assert _split_methods(" direct_path , ,rrt ") == ["direct_path", "rrt"]


def test_parse_plain_and_two_stage():
    assert _parse_method("rrt_connect") == ("rrt_connect", None, {})
    assert _parse_method("direct_path+stomp") == ("direct_path", "stomp", {})


def test_parse_overrides_are_typed():
    assert _parse_method("direct_path+stomp[w_obs=0,k_att=0.5,save_convergence_plot=false]") == (
        "direct_path", "stomp", {"w_obs": 0, "k_att": 0.5, "save_convergence_plot": False})


def test_parse_rejects_malformed_overrides():
    with pytest.raises(ValueError):
        _parse_method("direct_path+stomp[w_obs=0")
    with pytest.raises(ValueError):
        _parse_method("direct_path+stomp[w_obs]")


def test_safe_name_is_filesystem_safe():
    assert _safe_name("direct_path+stomp[w_obs=0,k_att=0]") == "direct_path+stomp_w_obs_0_k_att_0_"
