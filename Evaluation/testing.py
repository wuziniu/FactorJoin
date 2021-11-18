import pickle


def test_on_stats(model_path):
    with open(model_path, "rb") as f:
        bound_ensemble = pickle.load(f)

