import pytest
import numpy as np



def test_model_signature(staging_model,x_test_data):
    #check model has attribute predict
    assert hasattr(staging_model,'predict')
    # pred
    pred=staging_model.predict(x_test_data)
    # check outpur lenght is same as input len
    assert len(pred)==x_test_data.shape[0]
    # check output is 1D
    assert pred.ndim==1
    # check output is numeric or not
    np.issubdtype(pred.dtype,np.number)
    # checking there are no any nan prediction
    assert not np.isnan(pred).any()

   
