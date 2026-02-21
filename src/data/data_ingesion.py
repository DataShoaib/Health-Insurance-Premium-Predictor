import pandas as pd
import numpy as np
import os
from logger.logger import get_logger

logger=get_logger("Data_Ingesiom")


def data_load(data_path:str)->pd.DataFrame:
    try:
        logger.info(f"trying to load the data from the {data_path}")
        data=pd.read_csv(data_path)
        logger.info("Data loaded successfully")
        return data
    except FileNotFoundError as e:
        logger.error(f"File not found :{e}")
    except Exception as e:
        logger.error(f'an error accurred while loading the data:{e}')


def data_cleaning(data:pd.DataFrame)->pd.DataFrame:
    try:
        logger.info('data cleaning started')
        data.dropna(inplace=True)
        logger.info('null values dropped successfully')
        data.drop_duplicates(inplace=True)
        logger.info('drop duplicates values successfully')
        data.drop(columns=['Id'],inplace=True)
        logger.info('removed the id column succesfully')
        return data
    except Exception as e:
        logger.error(f"an error accured while cleaning the data:{e}")

def save_cleaned_data(data:pd.DataFrame,data_save_path:str)->None:
    try:
        logger.info(f"trying to save the cleaned data at: {data_save_path}")
        data_save_raw_path=os.path.join(data_save_path,'interim')
        os.makedirs(data_save_raw_path,exist_ok=True)
        data.to_csv(os.path.join(data_save_raw_path,'cleaned_data.csv'),index=False)
        logger.info(f'cleaned_data saved successfully at: {data_save_raw_path}')
    except Exception as e:
        logger.error(f'an error accured while saving the data :{e}')

def main():
    data=data_load('data/raw/insurance.csv') 
    if data is not None:
        cleaned_data=data_cleaning(data)
        if cleaned_data is not None:
           save_cleaned_data(cleaned_data,'data')

if __name__=="__main__":
    main()

        


  