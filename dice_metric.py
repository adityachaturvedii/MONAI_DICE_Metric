class DiceMetric(CumulativeIterationMetric):
    """
    Compute average Dice score for a set of pairs of prediction-groundtruth segmentations.

    It supports both multi-classes and multi-labels tasks.
    Input `y_pred` is compared with ground truth `y`.
    `y_pred` is expected to have binarized predictions and `y` can be single-channel class indices or in the
    one-hot format. The `include_background` parameter can be set to ``False`` to exclude
    the first category (channel index 0) which is by convention assumed to be background. If the non-background
    segmentations are small compared to the total image size they can get overwhelmed by the signal from the
    background. `y_preds` and `y` can be a list of channel-first Tensor (CHW[D]) or a batch-first Tensor (BCHW[D]),
    `y` can also be in the format of `B1HW[D]`.

    Example of the typical execution steps of this metric class follows :py:class:`monai.metrics.metric.Cumulative`.

    Args:
        include_background: whether to include Dice computation on the first channel of
            the predicted output. Defaults to ``True``.
        reduction: define mode of reduction to the metrics, will only apply reduction on `not-nan` values,
            available reduction modes: {``"none"``, ``"mean"``, ``"sum"``, ``"mean_batch"``, ``"sum_batch"``,
            ``"mean_channel"``, ``"sum_channel"``}, default to ``"mean"``. if "none", will not do reduction.
        get_not_nans: whether to return the `not_nans` count, if True, aggregate() returns (metric, not_nans).
            Here `not_nans` count the number of not nans for the metric, thus its shape equals to the shape of the metric.
        ignore_empty: whether to ignore empty ground truth cases during calculation.
            If `True`, NaN value will be set for empty ground truth cases.
            If `False`, 1 will be set if the predictions of empty ground truth cases are also empty.
        num_classes: number of input channels (always including the background). When this is None,
            ``y_pred.shape[1]`` will be used. This option is useful when both ``y_pred`` and ``y`` are
            single-channel class indices and the number of classes is not automatically inferred from data.
        return_with_label: whether to return the metrics with label, only works when reduction is "mean_batch".
            If `True`, use "label_{index}" as the key corresponding to C channels; if 'include_background' is True,
            the index begins at "0", otherwise at "1". It can also take a list of label names.
            The outcome will then be returned as a dictionary.

    """

    def __init__(
        self,
        include_background: bool = True,
        reduction: MetricReduction | str = MetricReduction.MEAN,
        get_not_nans: bool = False,
        ignore_empty: bool = True,
        num_classes: int | None = None,
        return_with_label: bool | list[str] = False,
    ) -> None:
        super().__init__()
        self.include_background = include_background
        self.reduction = reduction
        self.get_not_nans = get_not_nans
        self.ignore_empty = ignore_empty
        self.num_classes = num_classes
        self.return_with_label = return_with_label
        self.dice_helper = DiceHelper(
            include_background=self.include_background,
            reduction=MetricReduction.NONE,
            get_not_nans=False,
            softmax=False,
            ignore_empty=self.ignore_empty,
            num_classes=self.num_classes,
        )

    def _compute_tensor(self, y_pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """
        Args:
            y_pred: input data to compute, typical segmentation model output.
                It must be one-hot format and first dim is batch, example shape: [16, 3, 32, 32]. The values
                should be binarized.
            y: ground truth to compute mean Dice metric. `y` can be single-channel class indices or
                in the one-hot format.

        Raises:
            ValueError: when `y_pred` has less than three dimensions.
        """
        dims = y_pred.ndimension()
        if dims < 3:
            raise ValueError(f"y_pred should have at least 3 dimensions (batch, channel, spatial), got {dims}.")
        # compute dice (BxC) for each channel for each batch
        return self.dice_helper(y_pred=y_pred, y=y)  # type: ignore

    def aggregate(
        self, reduction: MetricReduction | str | None = None
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Execute reduction and aggregation logic for the output of `compute_dice`.

        Args:
            reduction: define mode of reduction to the metrics, will only apply reduction on `not-nan` values,
                available reduction modes: {``"none"``, ``"mean"``, ``"sum"``, ``"mean_batch"``, ``"sum_batch"``,
                ``"mean_channel"``, ``"sum_channel"``}, default to `self.reduction`. if "none", will not do reduction.

        """
        data = self.get_buffer()
        if not isinstance(data, torch.Tensor):
            raise ValueError(f"the data to aggregate must be PyTorch Tensor, got {type(data)}.")

        # do metric reduction
        f, not_nans = do_metric_reduction(data, reduction or self.reduction)
        if self.reduction == MetricReduction.MEAN_BATCH and self.return_with_label:
            _f = {}
            if isinstance(self.return_with_label, bool):
                for i, v in enumerate(f):
                    _label_key = f"label_{i+1}" if not self.include_background else f"label_{i}"
                    _f[_label_key] = round(v.item(), 4)
            else:
                for key, v in zip(self.return_with_label, f):
                    _f[key] = round(v.item(), 4)
            f = _f
        return (f, not_nans) if self.get_not_nans else f
