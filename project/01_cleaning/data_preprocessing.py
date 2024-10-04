# preprocess data
def clean_dataset(df, drop_null_rows, cols_to_drop=[]):
  '''
  df: dataframe
  drop_null_rows: boolean of whether or not to drop all rows with null values
  cols_to_drop: array of strings of columns to drop
  '''
  total_rows = df.index.size
  if len(cols_to_drop) > 0:
    df = df.drop(cols_to_drop, axis=1)

  total_null_rows = total_rows - df.dropna().index.size
  if drop_null_rows == True:
    df = df.dropna()

    print("Dropped " + str(round(total_null_rows/total_rows * 100, 2)) + "% of rows.")
  return df

