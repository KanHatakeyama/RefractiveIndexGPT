from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer
import numpy as np
from sklearn.cluster import KMeans
import copy
import pandas as pd
from boruta import BorutaPy
from sklearn.ensemble import RandomForestRegressor


class AutoBoruta:
    def __init__(self) -> None:
        self.desc_dict = {}

    def __call__(self, X, y):
        index = list(X.index)
        index = tuple(index)
        if index not in self.desc_dict:
            self.desc_dict[index] = select_features_by_boruta(X, y)
            print("Boruta selected features: ", index, self.desc_dict[index])

        return self.desc_dict[index]


def select_features_by_boruta(X, y):
    rf = RandomForestRegressor(n_jobs=-1)
    feat_selector = BorutaPy(rf, n_estimators='auto',
                             verbose=0, random_state=42)
    X = X.fillna(X.median())
    X_array = np.array(X).astype(float)
    feat_selector.fit(X_array, y.values.ravel())
    selected = X.columns[feat_selector.support_].to_list()
    return selected


def auto_evaluation(dataset_df,
                    model,
                    k=5,
                    random_CV=True,
                    selected_descriptors=[],
                    y_column_name="Experimental n",
                    autoboruta=None,
                    do_boruta=False):

    if len(selected_descriptors) > 0:
        dataset_df = copy.deepcopy(dataset_df)
        selected_descriptors = [
            i for i in selected_descriptors if i in dataset_df.columns]
        sel_df = dataset_df[selected_descriptors]
        sel_df[y_column_name] = dataset_df[y_column_name]
        sel_df["tags"] = dataset_df["tags"]
        dataset_df = sel_df

    train_df = dataset_df[dataset_df["tags"] != 7]
    X, y = prepare_X_and_y(train_df)

    if random_CV:
        clust_X = copy.deepcopy(X)
        np.random.seed(42)
        rows = len(X)
        numbers = np.array([i for i in range(k)] * (rows // k))
        # random numbers for remaining rows
        remaining_rows = rows % k
        remaining_numbers = np.random.choice(
            range(k), remaining_rows, replace=False)
        numbers = np.concatenate((numbers, remaining_numbers))
        np.random.shuffle(numbers)
        clust_X["Cluster"] = numbers
    else:
        clust_X = clusterize_X(X, k=k)

    y_ = pd.DataFrame(y, columns=["Experimental n"])
    y_["Cluster"] = clust_X["Cluster"]

    res_dict = {}
    r = cross_validation_df(clust_X, y_, model, k=k,
                            autoboruta=autoboruta, boruta=do_boruta)

    res_dict["CV_plot"] = [r[i]["y"] for i in range(k)]
    res_dict["CV_MAE"] = np.mean([r[i]["MAE"] for i in range(k)])
    res_dict["CV_RMSE"] = np.mean([r[i]["RMSE"] for i in range(k)])
    res_dict["CV_R2"] = np.mean([r[i]["R2"] for i in range(k)])

    # test
    """
    res = calc_test_score(dataset_df, model)
    scores = res[0]
    res_dict["TEST_MAE"] = scores["MAE"]
    res_dict["TEST_RMSE"] = scores["RMSE"]
    res_dict["TEST_R2"] = scores["R2"]
    res_dict["TEST_plot"] = res[1]
    res_dict["TEST_model"] = res[2]
    """
    return res_dict


def prepare_pipeline(model):
    return Pipeline([("imputer", SimpleImputer()), ("scaler", StandardScaler()), ("reg", model)])


def cross_validation_df(X, y, model, k=5, autoboruta=None, boruta=False):
    res_dict = {}
    model = prepare_pipeline(model)
    for i in range(k):
        test_X = (X[X['Cluster'] == i])
        train_X = (X[X['Cluster'] != i])
        test_y = (y[y['Cluster'] == i])
        train_y = (y[y['Cluster'] != i])

        test_y = test_y.drop("Cluster", axis=1)
        train_y = train_y.drop("Cluster", axis=1)
        train_X = train_X.drop("Cluster", axis=1)
        test_X = test_X.drop("Cluster", axis=1)

        if boruta:
            selected = autoboruta(train_X, train_y)
            train_X = train_X[selected]
            test_X = test_X[selected]

        model.fit(train_X, train_y)

        score_dict = {}
        pred_y = model.predict(test_X)
        score_dict["MAE"] = mean_absolute_error(test_y, pred_y)
        score_dict["R2"] = r2_score(test_y, pred_y)
        score_dict["RMSE"] = np.sqrt(
            mean_squared_error(test_y, pred_y))
        score_dict["y"] = (test_y, pred_y)

        res_dict[i] = score_dict

    return res_dict


def clusterize_X(X, k=5):

    pipe = Pipeline([("imputer", SimpleImputer()), ("scaler", StandardScaler(
    )), ("kmeans", KMeans(n_clusters=k, random_state=42))])

    X.fillna(X.mean(), inplace=True)
    clusters = pipe.fit_predict(X)

    X['Cluster'] = clusters

    return X


def prepare_X_and_y(dataset_df, y_colunm_name="Experimental n"):
    y = dataset_df[y_colunm_name]
    X = dataset_df.drop(y_colunm_name, axis=1)
    drop_list = ["tags", "title", "reference", "SMILES",
                 "Experimental_volume", "Experimental_alpha"]
    for c in drop_list:
        if c in X.columns:
            X = X.drop(c, axis=1)
    return X, y


def calc_test_score(dataset_df, model):
    train_df = dataset_df[dataset_df["tags"] != 7]
    test_df = dataset_df[dataset_df["tags"] == 7]

    tr_X, tr_y = prepare_X_and_y(train_df)
    te_X, te_y = prepare_X_and_y(test_df)

    pipe = prepare_pipeline(model)
    pipe.fit(tr_X, tr_y)
    pred_y = pipe.predict(te_X)
    scores = {}
    scores["R2"] = r2_score(te_y, pred_y)
    scores["MAE"] = mean_absolute_error(te_y, pred_y)
    scores["RMSE"] = mean_squared_error(te_y, pred_y, squared=False)
    return scores, (te_y, pred_y), pipe
