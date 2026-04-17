from curv_distribution_utils import _resolve_node_name

GRAPH = {
    "prev_down_proj": {"layer": 0, "prev": ["prev_up_proj", "prev_gate_proj"], "next": ["v_proj"], "prev_in": ["prev_o_proj"], "next_out": ["v_proj"]},
    "q_proj": {"layer": 1, "prev": ["prev_down_proj"], "next": ["k_proj"], "prev_in": ["prev_gate_up_out"], "next_out": ["A"]},
    "k_proj": {"layer": 1, "prev": ["prev_down_proj"], "next": ["q_proj"], "prev_in": ["prev_gate_up_out"], "next_out": ["A"]},
    "v_proj": {"layer": 1, "prev": ["prev_down_proj"], "next": ["A"], "prev_in": ["prev_gate_up_out"], "next_out": ["Att_out"]},
    "A": {"layer": 2, "prev": [], "next": ["v_proj"], "prev_in": [], "next_out": ["v_proj"]},
    "o_proj": {"layer": 3, "prev": ["A"], "next": ["gate_proj", "up_proj"], "prev_in": ["v_proj"], "next_out": ["gate_up_out"]},
    "gate_proj": {"layer": 4.1, "prev": ["o_proj"], "next": ["down_proj"], "prev_in": ["Att_out"], "next_out": ["down_proj"]},
    "up_proj": {"layer": 4.2, "prev": ["o_proj"], "next": ["down_proj"], "prev_in": ["Att_out"], "next_out": ["down_proj"]},
    "down_proj": {"layer": 5, "prev": ["up_proj", "gate_proj"], "next": ["v_proj"], "prev_in": ["o_proj"], "next_out": ["lm_head"]},
    "lm_head": {"layer": 6, "prev": ["down_proj"], "next": [], "prev_in": ["gate_up_out"], "next_out": []},
}


dim_to_layers = {
    0: ["prev_down_proj"],
    1: ["v_proj"],
    2: ["A"],
    3: ["o_proj"],
    4.1: ["gate_proj"],
    4.2: ["up_proj"],
    5: ["down_proj"],
    6: ["lm_head"],
}



def _resolve_graph_sets(operations, short_name):
    rel = GRAPH.get(short_name, {})

    prev_in_name = _resolve_node_name(operations, rel.get("prev_in", []))
    next_out_name = _resolve_node_name(operations, rel.get("next_out", []))
    prev_cost_names = rel.get("prev", [])
    next_cost_names = rel.get("next", [])

    return {
        "prev_in": operations.get(prev_in_name),
        "next_out": operations.get(next_out_name),
        "prev_in_name": prev_in_name,
        "next_out_name": next_out_name,
        "prev_cost_names": prev_cost_names,
        "next_cost_names": next_cost_names,
    }
