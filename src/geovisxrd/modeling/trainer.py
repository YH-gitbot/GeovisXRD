from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
from .model_factory import get_model

def train_model(
    model_name,
    X,
    y,
    test_size=0.2,
    random_state=1,
    return_metrics=True,
    **model_params
):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    model = get_model(model_name, **model_params)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    rmse = root_mean_squared_error(y_test, y_pred)

    if return_metrics:
        return model, {"rmse": rmse, "y_pred": y_pred}
    else:
        return model
