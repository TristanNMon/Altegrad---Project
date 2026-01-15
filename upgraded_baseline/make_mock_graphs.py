import pickle


def make_small(in_path, out_path, n):
    with open(in_path, "rb") as f:
        data = pickle.load(f)
    with open(out_path, "wb") as f:
        pickle.dump(data[:n], f)


if __name__ == "__main__":
    make_small("../data/train_graphs.pkl", "../data/train_graphs_mock.pkl", 32)
    make_small("../data/validation_graphs.pkl", "../data/val_graphs_mock.pkl", 16)
    make_small("../data/test_graphs.pkl", "../data/test_graphs_mock.pkl", 16)
