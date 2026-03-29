GRAPH_RELATION = {

    # =========================
    # ATTENTION BLOCK
    # =========================
    "q_proj": {
        "prev": ["prev_down_proj"],
        "next": ["A_q"],
        "next_cost": ["k_proj"]
    },

    "k_proj": {
        "prev": ["prev_down_proj"],
        "next": ["A_k"],
        "next_cost": ["q_proj"]
    },

    "v_proj": {
        "prev": ["prev_down_proj"],
        "next": ["A"],
    },

    "A": {
        "prev": ["v_proj"],
        "next": ["o_proj"]
    },

    "o_proj": {
        "prev": ["A"],
        "next": ["gate_proj", "up_proj"]
    },

    # =========================
    # MLP BLOCK
    # =========================
    "gate_proj": {
        "prev": ["o_proj"],
        "next": ["down_proj"]
    },

    "up_proj": {
        "prev": ["o_proj"],
        "next": ["down_proj"]
    },

    # =========================
    # DEFERRED CROSS-LAYER EDGE
    # =========================
    "down_proj": {
        "prev": ["down_proj_in"],
        "prev_cost": ["gate_proj", "up_proj"],
        "next": ["lm_head"]
    },

    # This is the deferred version used in next layer
    "prev_down_proj": {
        "prev": ["prev_down_proj_in"],
        "prev_cost": ["prev_gate_proj", "prev_up_proj"],
        "next": ["q_proj", "k_proj", "v_proj"]
    },

    # =========================
    # FINAL OUTPUT
    # =========================
    "lm_head": {
        "prev": ["down_proj"],
        "next": []
    }
}
