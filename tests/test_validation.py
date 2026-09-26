from stocklab.validation import expanding_window_splits


def test_walk_forward_splits_are_chronological_and_non_overlapping():
    folds = list(
        expanding_window_splits(
            30,
            min_train_size=10,
            validation_size=3,
            test_size=4,
            step_size=4,
            gap=1,
        )
    )

    assert len(folds) == 3
    first = folds[0]
    assert first.train == slice(0, 10)
    assert first.validation == slice(10, 13)
    assert first.test == slice(14, 18)

    for fold in folds:
        assert fold.train.stop <= fold.validation.start
        assert fold.validation.stop < fold.test.start
