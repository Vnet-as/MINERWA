import argparse
import os
import pickle
import sys

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

# Append absolute path to of the 'src' directory to path
sys.path.append(os.path.dirname(os.path.realpath(__file__)).rsplit('src', 1)[0])

import src.utils as utils


DEFAULT_N_JOBS = 2
DEFAULT_CSV_SEPARATOR = ','
DEFAULT_RANDOM_STATE = -1


COLUMNS_TO_BE_REMOVED = [
    'IP_SRC', 'IPV4_SRC_ADDR', 'IPV6_SRC_ADDR',
    'IP_DST', 'IPV4_DST_ADDR', 'IPV6_DST_ADDR',
    'SRC_DENY', 'DST_DENY']


def main(raw_args: list) -> None:
    args = parse_args(raw_args)

    if args.random_state == -1:
        args.random_state = None

    train = utils.load_df(args.input_path, csv_sep=args.csv_sep)
    train_x = train.drop(columns=['Label'])
    train_x.drop(columns=COLUMNS_TO_BE_REMOVED, inplace=True, errors='ignore')
    train_x.drop(columns=['Label_source'], inplace=True, errors='ignore')
    train_y = list(train['Label'])

    if args.tune:
        # Random forest hyperparameter tuning & classifier training
        random_grid = {
            'n_estimators': [100, 200],  # number of trees in the random forest
            'criterion': ['gini', 'entropy'],  # function to measure the quality of a split
            'max_features': ['sqrt', 'log2', None],  # number of features in consideration at every split
            'max_depth': [None, 10, 100],  # maximum number of levels allowed in each decision tree
            'min_samples_split': [2, 6],  # minimum sample number to split a node
            'min_samples_leaf': [1, 3],  # minimum sample number that can be stored in a leaf node
            'bootstrap': [True, False],  # method used to sample data points
        }

        clf = RandomForestClassifier(random_state=0, n_jobs=args.n_jobs)
        clf_random = RandomizedSearchCV(estimator=clf, param_distributions=random_grid, n_iter=100,
                                        cv=5, verbose=2, random_state=args.random_state, n_jobs=1)
        clf_random.fit(train_x, train_y)

        print('Best hyperparameters:')
        print(clf_random.best_params_)

        clf = RandomForestClassifier(
            **clf_random.best_params_, random_state=args.random_state, n_jobs=args.n_jobs)
    else:
        clf = RandomForestClassifier(random_state=args.random_state, n_jobs=args.n_jobs, n_estimators=200,
                                     min_samples_split=6, min_samples_leaf=1, max_features='sqrt',
                                     max_depth=None, criterion='entropy', bootstrap=False)

    clf.fit(train_x, train_y)

    os.makedirs(os.path.dirname(args.model_filepath), exist_ok=True)

    with open(args.model_filepath, 'wb') as f:
        pickle.dump(clf, f)


def parse_args(raw_args: list) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Trains a classification-based filter model based on known attacks.')

    parser.add_argument(
        'input_path',
        help='Input file path containing known attacks.',
    )
    parser.add_argument(
        'model_filepath',
        help='File path containing trained filter model.',
    )

    parser.add_argument(
        '--tune',
        action=argparse.BooleanOptionalAction,
        help='If specified, perform hyperparameter tuning during model training.',
    )
    parser.set_defaults(tune=True)

    parser.add_argument(
        '--csv-sep',
        default=DEFAULT_CSV_SEPARATOR,
        help=('Separator to use for input and output files if the file format is CSV or gzip.'
              f' Default: {DEFAULT_CSV_SEPARATOR}'),
    )
    parser.add_argument(
        '--n-jobs',
        type=int,
        default=DEFAULT_N_JOBS,
        help=('Number of jobs to run in parallel during classifier training.'
              f' Default: {DEFAULT_N_JOBS}'),
    )
    parser.add_argument(
        '--random-state',
        default=DEFAULT_RANDOM_STATE,
        help=('Fixed random state to use for model training. Use -1 to not use a fixed random state.'
              f' Default: {DEFAULT_RANDOM_STATE}'),
    )

    return parser.parse_args(raw_args)


if __name__ == '__main__':
    main(sys.argv[1:])
