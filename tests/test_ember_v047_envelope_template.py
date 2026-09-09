from jobs import ember_envelope_template_v047 as v047


def test_config_is_v047():
    cfg = v047.load_config(v047.DEFAULT_CONFIG)
    assert cfg["version"] == "0.0.47"
    assert cfg["training_authorized"] is False
    assert cfg["gpu_training_authorized"] is False
    assert cfg["placement_learning_authorized"] is False


def test_residual_cohort_is_frozen():
    current = v047.residual_cases()
    prior = v047.v045.target_cases()
    assert len(current) == len(prior) == 24
    assert [(c["id"], c["prompt"]) for c in current] == [(c["id"], c["prompt"]) for c in prior]
