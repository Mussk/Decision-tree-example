import numpy as np

def get_Gini(y):
    instances = np.bincount(y)
    total = np.sum(instances)
    return 1.0 - np.sum(np.power(instances/total,2))

def get_Entropy(y):
    instances = np.bincount(y)
    total = np.sum(instances)
    p = instances / total
    return 0.0 - np.sum(np.log(p)*p)

def get_class_for_node(y):
    instances = np.bincount(y)
    return np.argmax(instances, axis=0)

def create_child_nodes(x, y, feature, threshold):
    x_l = []
    y_l = []
    x_r = []
    y_r = []
    for features, classification in zip(x,y):
        if features[feature] <= threshold:
            x_l.append(features)
            y_l.append(classification)
        else:
            x_r.append(features)
            y_r.append(classification)
    return np.asarray(x_l), np.asarray(y_l, dtype=np.int64), np.asarray(x_r), np.asarray(y_r, dtype=np.int64)

def get_score(y, y_l, y_r, impurity_measure):
    score_left = impurity_measure(y_l)*y_l.shape[0]/y.shape[0]
    score_right = impurity_measure(y_r)*y_r.shape[0]/y.shape[0]
    return score_left + score_right

def split_node_node(x, y, granulation, impurity_measure):
    x_l_best = None
    y_l_best = None
    x_r_best = None
    y_r_best = None
    score_best = None
    feature_best = None
    threshold_best = None
    for feature in range(x.shape[1]):
        start = np.min(x[:,feature])
        end = np.max(x[:,feature])
        step = (end - start) / granulation
        for threshold in np.arange(start, end, step):
            x_l, y_l, x_r, y_r = create_child_nodes(x, y, feature, threshold)
            score = get_score(y, y_l, y_r, impurity_measure)
            #print('{} - {} => {}'.format(x_names[feature], threshold, score))
            if score_best is None or score < score_best:
                x_l_best = x_l
                y_l_best = y_l
                x_r_best = x_r
                y_r_best = y_r
                score_best = score
                feature_best = feature
                threshold_best = threshold
    return x_l_best, y_l_best, x_r_best, y_r_best, score_best, feature_best, threshold_best