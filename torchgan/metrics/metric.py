__all__ = ['EvaluationMetric']

class EvaluationMetric(object):
    r"""
    Base class for all Evaluation Metrics
    """
    def preprocess(self, x):
        r"""
        Subclasses must override this function and provide their own preprocessing
        pipeline.

        :raises NotImplementedError: If the subclass doesn't override this function.
        """
        raise NotImplementedError

    def calculate_score(self, x):
        r"""
        Subclasses must override this function and provide their own score calculation.

        :raises NotImplementedError: If the subclass doesn't override this function.
        """
        raise NotImplementedError

    def __call__(self, x):
        return self.calculate_score(self.preprocess(x))

    def metric_ops(self, generator, discriminator, **kwargs):
        r"""
        Subclasses must override this function and provide their own metric evaluation ops.

        :raises NotImplementedError: If the subclass doesn't override this function.
        """
        raise NotImplementedError
