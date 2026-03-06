from sklearn.metrics import mean_absolute_error,r2_score,root_mean_squared_error

TOLERANCE = 0.05

def test_model_performance_and_comparing_with_prod(staging_model,production_model,x_test_data,y_test_data):
    y_pred=staging_model.predict(x_test_data)

    staging_mae=mean_absolute_error(y_test_data,y_pred)
    staging_r2=r2_score(y_test_data,y_pred)
    staging_rmse=root_mean_squared_error(y_test_data,y_pred)

    assert staging_mae >= 0
    assert staging_rmse >= 0
    assert -1 <= staging_r2 <= 1

    if production_model:

        prod_pred=production_model.predict(x_test_data)
        production_mae=mean_absolute_error(y_test_data,prod_pred)
        production_rmse=root_mean_squared_error(y_test_data,prod_pred)
        production_r2_scr=r2_score(y_test_data,prod_pred)

        assert staging_mae <= production_mae * (1 + TOLERANCE)
        assert staging_rmse <= production_rmse * (1 + TOLERANCE)
        assert staging_r2 >= production_r2_scr * (1 - TOLERANCE)