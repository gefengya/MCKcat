import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def calculate_metrics(file_path):
    """
    This function is used to read the CSV file and calculate the evaluation metrics between predicted and actual results.

    Parameters:
    file_path (str): Path to the CSV file

    Returns:
    None
    """
    # Read the CSV file
    data = pd.read_csv(file_path)

    # Extract prediction columns of 10 models
    model_predictions = data['pred_log10[kcat(s^-1)]']

    # Calculate the average of predictions from 10 models
    average_predictions = model_predictions

    # Extract actual result columns
    # actual_results = data['log10kcat_max']
    actual_results = data['true_log10[kcat(s^-1)]']

    # Calculate Pearson correlation coefficient
    correlation, p_value = pearsonr(actual_results, average_predictions)

    # Calculate Mean Squared Error (MSE)
    mse = mean_squared_error(actual_results, average_predictions)

    # Calculate Mean Absolute Error (MAE)
    mae = mean_absolute_error(actual_results, average_predictions)

    # Calculate Coefficient of Determination (R^2)
    r_squared = r2_score(actual_results, average_predictions)

    print(f"File Path: {file_path}")
    print(f"Pearson Correlation Coefficient (r): {correlation}")
    print(f"p-value: {p_value}")
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"Coefficient of Determination (R^2): {r_squared}")
    print("-" * 50)


if __name__ == "__main__":
    # Process CSV files named with single numbers
    for i in range(1, 6):
         file_path = f'/mnt/usb3/code/gfy/code/CataPro-master/models/kcat_models/splits/{i}/catapro_turnup.csv'
         calculate_metrics(file_path)
         file_path = f'/mnt/usb3/code/gfy/code/CataPro-master/models/kcat_models/splits_enzyme/{i}/catapro_turnup.csv'
         calculate_metrics(file_path)
         file_path = f'/mnt/usb3/code/gfy/code/CataPro-master/models/kcat_models/splits_kcat/{i}/catapro_turnup.csv'
         calculate_metrics(file_path)


