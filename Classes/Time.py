import pandas as pd

print('Model,Mean Prediction Time,Total Time (for 1,000 complaints)')
def sum_and_average_column(dataset: str, column: str) -> tuple:
    df = pd.read_csv(dataset)
    total = round(df[column].sum(), 2)
    average = round(df[column].mean(), 2)
    return column + ',' + str(average) + ',' + str(total)
    # return column, total, average

# print(sum_and_average_column("../Datasets/dataset.csv", "time-gpt-4o"))
# print(sum_and_average_column("../Datasets/dataset.csv", "time-gpt-4.5-preview-2025-02-27"))
# print(sum_and_average_column("../Datasets/dataset.csv", "time-o1-2024-12-17"))
# print(sum_and_average_column("../Datasets/dataset.csv", "time-gpt-4o-mini"))
# print(sum_and_average_column("../Datasets/dataset.csv", "time-o3-mini-2025-01-31"))
# print(sum_and_average_column("../Datasets/dataset.csv", "time-claude-3-5-sonnet-latest"))
# print(sum_and_average_column("../Datasets/dataset.csv", "time-claude-3-7-sonnet-latest"))
# print(sum_and_average_column("../Datasets/dataset.csv", "time-claude-3-5-haiku-latest"))
# print(sum_and_average_column("../Datasets/dataset.csv", "time-deepseek-chat"))
# print(sum_and_average_column("../Datasets/dataset.csv", "time-deepseek-reasoner"))
# print(sum_and_average_column("../Datasets/dataset.csv", "time-gemini-2.0-flash"))
# print(sum_and_average_column("../Datasets/dataset.csv", "time-gemini-2.0-flash-lite"))
# print(sum_and_average_column("../Datasets/dataset.csv", "time-gemini-1.5-pro"))
# print(sum_and_average_column("../Datasets/dataset.csv", "time-gemini-1.5-flash"))
