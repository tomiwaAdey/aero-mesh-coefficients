import numpy as np

from aero_mesh.classical_models import model_catalog, predict_class_probabilities


def test_probability_columns_follow_fixed_semantic_class_order():
    rng = np.random.default_rng(13)
    features = rng.normal(size=(80, 5))
    labels = np.tile(np.arange(4), 20)
    model = model_catalog(13)["random_forest"].fit(features, labels)
    probabilities = predict_class_probabilities(model, features)
    assert probabilities.shape == (80, 4)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
