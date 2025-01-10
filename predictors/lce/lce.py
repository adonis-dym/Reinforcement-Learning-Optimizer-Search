# This code is mostly from https://github.com/automl/pylearningcurvepredictor
# pylearningcurvepredictor author: Tobias Domhan, tdomhan

import numpy as np
from globals import TOTAL_EPOCHS
from predictors.predictor import Predictor
from predictors.lce.parametric_model import (
    model_name_list,
    model_config,
    construct_parametric_model,
)
from predictors.lce.parametric_ensemble import ParametricEnsemble


from typing import List


class LCEPredictor(Predictor):
    def __init__(self, metric=None):
        self.metric = metric

    def query(self, learning_curves: List[float]) -> float:
        # Construct the ensemble
        ensemble = ParametricEnsemble(
            [construct_parametric_model(model_config, name)
             for name in model_name_list]
        )

        learning_curves = np.array(learning_curves)
        # N为mcmc在后验概率分布中采样次数
        ensemble.mcmc(learning_curves, N=1000)
        # 在采样之后，我们可以一次传入多个epoch来让模型预测在这些里面每一个epoch上的准确率
        prediction = ensemble.mcmc_sample_predict([TOTAL_EPOCHS])
        prediction = np.squeeze(prediction)

        return prediction

    def get_data_reqs(self):
        """
        Returns a dictionary with info about whether the predictor needs
        extra info to train/query.
        """
        reqs = {
            "requires_partial_lc": True,
            "metric": self.metric,
            "requires_hyperparameters": False,
            "hyperparams": None,
            "unlabeled": False,
            "unlabeled_factor": 0,
        }
        return reqs


if __name__ == "__main__":
    print("Testing LCEPredictor")
    predictor = LCEPredictor(metric="accuracy")
    # 注：我们这里修改后的LCEPredictor使用accuracy的绝对值（0-1），而不是原来实现中的百分比（0-100）。输出也是0-1之间的值。
    # Suppose early_epochs_accuracies is your list of validation accuracies for the early epochs
    # learning_curves = [0.6, 0.65, 0.7, 0.72, 0.75,
    #                    0.76, 0.77, 0.78, 0.79, 0.8, 0.802, 0.803]
    # When the "point likelihood" is zero, it means that the probability of the data given the parameters at that point is zero. In other words, the current model parameters are extremely unlikely to produce the observed data.
    # 一种情况是数据太少，模型没有充分的信息拟合
    # 另一种情况是给定的数据不符合模型的趋势，e.g.看下面例子
    learning_curves = [0.1]*5
    # Now you can use the predictor to get the result
    predictions = predictor.query(learning_curves)
    print(predictions)
