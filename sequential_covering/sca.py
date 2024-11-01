import numpy as np
from sklearn.metrics import fbeta_score
from rule import Rule

class SequentialCovering(object):
    def __init__(self, min_coverage=0.01, max_conditions=5, beta=1):
        self.rules = []  # List of rules
        self.min_coverage = min_coverage  # Minimum coverage threshold
        self.max_conditions = max_conditions  # Maximum number of conditions for a rule
        self.beta = beta  # Beta value for F-beta score

    def fit(self, X, y):
        covered = np.zeros(len(X), dtype=bool)  # Keeps track of covered instances
        
        # While uncovered instances exist
        while np.sum(~covered) > 0:
            rule = self.learn_rule(X, y, covered)
            
            if rule is None: 
                break  # If no more valid rules can be found
            
            self.rules.append(rule)
            covered |= self.apply_rule(rule, X)  # Mark instances as covered by rule

    def learn_rule(self, X, y, covered):
        """
        Learn a single rule with multiple conditions that covers uncovered instances.
        """
        conditions = []
        best_rule = None
        best_fbeta = 0

        while len(conditions) < self.max_conditions:
            best_condition = None
            best_condition_fbeta = 0

            for feature_idx in range(X.shape[1]):  # Iterate over features
                for value in np.unique(X[:, feature_idx]):
                    candidate_conditions = conditions + [(feature_idx, value)]
                    rule = Rule(conditions=candidate_conditions, target_class=self.majority_class(y, ~covered))
                    
                    # Evaluate the candidate rule using fbeta_score
                    fbeta, coverage = self.evaluate_rule(rule, X, y, covered)

                    # Select the best condition to add
                    if coverage >= self.min_coverage and fbeta > best_condition_fbeta:
                        best_condition_fbeta = fbeta
                        best_condition = (feature_idx, value)

            if best_condition is not None:
                conditions.append(best_condition)
                rule = Rule(conditions=conditions, target_class=self.majority_class(y, ~covered))
                best_fbeta = best_condition_fbeta
                best_rule = rule
            else:
                break  # No more improvements can be made

        return best_rule if best_rule else None

    def majority_class(self, y, mask):
        """ Return the majority class for the current instances. """
        return np.argmax(np.bincount(y[mask]))

    def apply_rule(self, rule, X):
        """ Apply rule to instances and return a boolean mask of covered instances """
        return np.array([rule.covers(instance) for instance in X])
    
    def evaluate_rule(self, rule, X, y, covered):
        """ Evaluate the F-beta score and coverage of a rule """
        covered_instances = np.array([rule.covers(instance) for instance in X])
        uncovered_instances = ~covered  # Focus on currently uncovered instances
        
        if np.sum(covered_instances & uncovered_instances) == 0:
            return 0, 0  # No uncovered instances are covered by the rule

        y_true = y[uncovered_instances]  # True labels for uncovered instances
        y_pred = covered_instances[uncovered_instances] * rule.target_class  # Predictions for uncovered instances
        
        # Calculate F-beta score (need at least one positive and one negative sample)
        fbeta = fbeta_score(y_true, y_pred, average='binary', beta=self.beta, zero_division=0)
        
        coverage = np.sum(covered_instances) / len(X)  # Proportion of the dataset covered by the rule
        
        return fbeta, coverage

    def predict(self, X):
        predictions = []
        for instance in X:
            for rule in self.rules:
                if rule.covers(instance):
                    predictions.append(rule.target_class)
                    break
            else:
                predictions.append(None)  # No rule applies
        return predictions