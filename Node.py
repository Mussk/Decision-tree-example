class Node:

    is_leaf = False
    data = None
    target = None
    score = None
    feature = None
    threshold = None
    child_nodes = None

    def __init__(self,data,target,feature_names,target_names):
        self.data = data
        self.target = target
        self.feature_names = feature_names
        self.target_names = target_names

    def __str__(self):
        return print("Is leaf: {} \nData: {} \nTarget: {}\nScore: {}\nFeature: {}\nThreshold: {}\nChild_nodes: {}\n"
                     .format(self.is_leaf,self.data,self.target,self.score,self.feature,self.threshold,self.child_nodes))
