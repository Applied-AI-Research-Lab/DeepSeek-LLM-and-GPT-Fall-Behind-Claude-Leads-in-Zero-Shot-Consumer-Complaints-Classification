import pandas as pd


def sum_and_average_column(dataset: str, column: str) -> tuple:
    df = pd.read_csv(dataset)
    total = round(df[column].sum(), 2)
    average = round(df[column].mean(), 2)
    return column, total, average

print(sum_and_average_column("../Datasets/dataset.csv", "time-gpt-4o"))
print(sum_and_average_column("../Datasets/dataset.csv", "time-gpt-4o-mini"))
print(sum_and_average_column("../Datasets/dataset.csv", "time-claude-3-5-sonnet-latest"))
print(sum_and_average_column("../Datasets/dataset.csv", "time-claude-3-5-haiku-latest"))
print(sum_and_average_column("../Datasets/dataset.csv", "time-deepseek-chat"))

