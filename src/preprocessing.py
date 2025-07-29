def split_features_labels(data):
    X = data.drop(columns=['specific.disorder'])
    y = data['specific.disorder']
    return X, y
