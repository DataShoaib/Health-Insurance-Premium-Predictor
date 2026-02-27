import pandas as pd
import numpy as np
from logger.logger import get_logger
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder,OneHotEncoder,StandardScaler,FunctionTransformer
import os

logger=get_logger('Feature-Engineering')

def load_cleaned_data(data_path:str)->pd.DataFrame:
    try:
        logger.info(f'loading data for Feature-eng at : {data_path}')
        cleaned_data=pd.read_csv(data_path)
        logger.info(f'data loaded successfully at :{data_path}')
        return cleaned_data
    except FileNotFoundError as e:
        logger.error(f'File not found from the {data_path}:{e}')
        raise
    except Exception as e:
        logger.error(f'an error accured while loading the cleaned data:{e}')
        raise


def feature_Construction(data:pd.DataFrame)->pd.DataFrame:
    try:
        logger.info('feature engineering started')
        data['bmi_group']=pd.cut(data['bmi'],bins=[0,18.5,25,30,100],labels=['underweight','normal','overweight','obese'])
        logger.info('bmi_group feature created successfully')
        data['high_bmi_risk']=data['bmi_group'].apply(lambda x:1 if x=='obese' else 0)
        logger.info('high_bmi_risk feature created successfully')
        data['high_bp_risk']=data['bloodpressure'].apply(lambda x:1 if x>100 else 0)     
        logger.info('high_bp_risk feature created successfully')
        data['age_risk'] = pd.cut(data['age'],bins=[17,30,45,60],labels=['Low','Medium','High'])
        logger.info('age_risk feature created successfully')
        data['smoker_risk']=data['smoker'].apply(lambda x:1 if x=='Yes' else 0)
        logger.info('smoker_risk feature created successfully')
        data['risk_score']=data['high_bmi_risk']+data['high_bp_risk']+(data['age']>45).astype(int)
        logger.info('risk_score feature created successfully')
        logger.info('feature engineering completed successfully')
        logger.info(f'data feature:{data.columns}')
        return data
    except Exception as e:
        logger.error(f'an error accured while feature engineering the data:{e}')
        raise


def spliting_data(data:pd.DataFrame)->tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    try:
        logger.info("Splitting data into X and y")
        X = data.drop('claim', axis=1)
        y = data['claim']

        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        logger.info("Train-test split completed")
        return x_train,x_test,y_train,y_test
    except Exception as e:
        logger.error(f'an error accured while the train test split')
        raise

def save_split_data(x_train:pd.DataFrame,x_test:pd.DataFrame,y_train:pd.Series,y_test:pd.Series,save_data_path:str)->None:
        try:
            logger.info(f'saving the split data at :{save_data_path}')        
            save_data_final_path=os.path.join(save_data_path,'proccessed')
            os.makedirs(save_data_final_path,exist_ok=True)
            x_train.to_csv(os.path.join(save_data_final_path,'x_train.csv'),index=False)
            x_test.to_csv(os.path.join(save_data_final_path,'x_test.csv'),index=False)
            y_train.to_csv(os.path.join(save_data_final_path,'y_train.csv'),index=False)
            y_test.to_csv(os.path.join(save_data_final_path,'y_test.csv'),index=False)
            logger.info(f'split data saved successfully at :{save_data_final_path}')
        except Exception as e:
            logger.error(f'an error accured while saving the split data:{e}')
            raise

def binary_label_encoder(x):
            return np.where(x == 'Yes', 1, 0)            


def pipeline_preprocessor():
    try:
        logger.info('creating preprocessing for the model Pipeline')
        

        binary_tf = FunctionTransformer(
            binary_label_encoder,
            validate=False,
            feature_names_out='one-to-one'
        )

        ordinal_encoder = OrdinalEncoder(
            categories=[
                ['Low', 'Medium', 'High'],
                ['underweight', 'normal', 'overweight', 'obese']
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), ['age', 'bmi', 'bloodpressure']),
                ('ohe', OneHotEncoder(drop='first', handle_unknown='ignore'), ['region']),
                ('ordinal', ordinal_encoder, ['age_risk', 'bmi_group']),
                ('binary', binary_tf, ['smoker', 'diabetic', 'gender', 'smoker_risk'])
            ],
            remainder='passthrough'
        )

        logger.info("Preprocessing pipeline created successfully")

        return preprocessor

    except Exception as e:
        logger.exception(f"Error while creating preprocessing pipeline: {e}")
        raise
    
def main():
    cleaned_data = load_cleaned_data('data/interim/cleaned_data.csv')
    feature_created_data = feature_Construction(cleaned_data)
    x_train, x_test, y_train, y_test = spliting_data(feature_created_data)
    save_split_data(x_train, x_test, y_train, y_test, 'data')
if __name__=="__main__":
    main()