class Rule(object):
    def __init__(self, conditions, target_class):
        self.conditions = conditions  # List of tuples (attribute, value); can convert to Named tuples for each class
        self.target_class = target_class

    def covers(self, instance):
        for attribute, value in self.conditions:
            if instance[attribute] != value:
                return False
        return True