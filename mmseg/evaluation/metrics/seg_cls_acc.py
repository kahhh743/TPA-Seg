from typing import Sequence, List

from mmengine.evaluator import BaseMetric
from mmengine.registry import METRICS

import numpy as np


@METRICS.register_module()  # 将 Accuracy 类注册到 METRICS 注册器
class SegClsAcc(BaseMetric):
    """ Accuracy Evaluator

    Default prefix: ACC

    Metrics:
        - accuracy (float): classification accuracy
    """

    default_prefix = 'SegClsAcc'  # 设置 default_prefix

    def process(self, data_batch: Sequence[dict], data_samples: Sequence[dict]):
        """Process one batch of data and predictions. The processed
        Results should be stored in `self.results`, which will be used
        to compute the metrics when all batches have been processed.

        Args:
            data_batch (Sequence[Tuple[Any, dict]]): A batch of data
                from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from
                the model.
        """

        # 取出分类预测结果和类别标签
        str = ''
        result = {
            'pred': data_samples['pred_label'],
            'gt': data_samples['data_sample']['gt_label']
        }

        # 将当前 batch 的结果存进 self.results
        self.results.append(result)

    def compute_metrics(self, results: List):
        """Compute the metrics from processed results.

        Args:
            results (dict): The processed results of each batch.

        Returns:
            Dict: The computed metrics. The keys are the names of the metrics,
            and the values are corresponding results.
        """

        # 汇总所有样本的分类预测结果和类别标签
        test = ''

        preds = np.concatenate([res['pred'] for res in results])
        gts = np.concatenate([res['gt'] for res in results])

        # 计算分类正确率
        acc = (preds == gts).sum() / preds.size

        # 返回评测指标结果
        return {'accuracy': acc}