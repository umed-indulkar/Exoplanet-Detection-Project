import math

class LockedHyperparams:
    def __init__(self, input_features):
        self._input_dim = input_features
        self._embedding_dim = max(2, round(math.sqrt(input_features * ((2*input_features)//3//2))))
        self._hidden1 = max(4, round(2/3 * input_features + self._embedding_dim))
        self._hidden2 = max(2, round(self._hidden1 / 2))
        self._classifier_hidden = max(2, round(self._hidden2 / 2))

        self._locked = True

    @property
    def INPUT_DIM(self):
        return self._input_dim
    @property
    def HIDDEN1(self):
        return self._hidden1
    @property
    def HIDDEN2(self):
        return self._hidden2
    @property
    def EMBEDDING_DIM(self):
        return self._embedding_dim
    @property
    def CLASSIFIER_HIDDEN(self):
        return self._classifier_hidden

    def __setattr__(self, key, value):
        if hasattr(self, '_locked') and self._locked and key in ['_input_dim', '_hidden1', '_hidden2', '_embedding_dim', '_classifier_hidden']:
            raise AttributeError(f"{key} is locked and cannot be modified!")
        super().__setattr__(key, value)