import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from scikit_wrapRs.wrap_r_ranger import RangerClassifier


########################################################################################################################
# Grab data and save base
data, target = load_breast_cancer(return_X_y=True)
train_x, test_x, target_x, target_y = train_test_split(data, target, test_size=0.1, random_state=42)
X = pd.DataFrame(train_x)
y = pd.DataFrame(target_x, columns=['target'])
test_x = pd.DataFrame(test_x)
test_y = pd.DataFrame(target_y, columns=['target'])
########################################################################################################################

breast_cancer = RangerClassifier(num_trees=500, num_threads=8, seed=42, verbose=True)
breast_cancer.fit(X, y)
breast_cancer.predict_proba(test_x)
