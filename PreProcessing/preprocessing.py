import numpy as np
import pandas as pd


def _change_type(polyline):
    polyline = polyline[2:-2]
    cords_raw = polyline.split("],[")
    cords = []
    max_lon, min_lon, max_lat, min_lat = float("-inf"), float("inf"), float("-inf"), float("inf")
    for cord in cords_raw:
        cords.append([float(loc) for loc in cord.split(",")])
        max_lon = max(max_lon, cords[-1][0])
        min_lon = min(min_lon, cords[-1][0])
        max_lat = max(max_lat, cords[-1][1])
        min_lat = min(min_lat, cords[-1][1])
    return pd.Series({"POLYLINE": cords, "max_lon": max_lon, "min_lon": min_lon, "max_lat": max_lat, "min_lat": min_lat})

def filter_map(train, max_lat, min_lat, max_long, min_long):
    #changed = train["POLYLINE"].apply(_change_type)
    train.reset_index(drop=True,inplace=True)
    train["check"] = [0 for i in range(len(train))]
    for i in range(len(train)):
        for cord in train.iloc[i]['POLYLINE']:
            if cord[0] < max_long and cord[0] > min_long and cord[1] < max_lat and cord[1] > min_lat:
                train["check"].iloc[i] = 1
            else:
                train["check"].iloc[i] = 0
    return train[train["check"] == 1]


def _normalize(polyline, max_lon, min_lon, max_lat, min_lat):
    final = [[(cord[0]-min_lon)/(max_lon-min_lon), (cord[1] - min_lat) / (max_lat - min_lat)] for cord in polyline]
    return pd.Series({"POLYLINE_INIT": final[:-1], "POLYLINE_DEST": final[-1]})


def _to_matrix(polyline, m):
    mat = np.zeros((m, m))
    n = len(polyline)
    for i in range(n):
        x = min(m - 1, int(polyline[i][0] * m))
        y = min(m - 1, int(polyline[i][1] * m))
        mat[y][x] = (i + 1) / n
    return mat


def transform(df_train, m):
    # Change type
    changed = df_train["POLYLINE"].apply(_change_type)
    # Filter map for max/min long/lat
    changed = filter_map(changed,41.5,40,-8.6,-9)
    #df_train["POLYLINE"] = changed["POLYLINE"] #要改
    # Get min-max
    max_longitude = changed["max_lon"].max()
    min_longitude = changed["min_lon"].min()
    max_latitude = changed["max_lat"].max()
    min_latitude = changed["min_lat"].min()
    # Normalize min-max and split
    cleaned = changed["POLYLINE"].apply(_normalize, args=(max_longitude, min_longitude, max_latitude, min_latitude))
    # Transform to matrices
    cleaned["MATRIX"] = cleaned["POLYLINE_INIT"].apply(_to_matrix, args=(m,))
    return cleaned


if __name__ == "__main__":
    train = pd.read_csv("/Users/xinchengzhu/Downloads/train.csv")
    # Filter missing data and useless columns
    train = train[train["MISSING_DATA"] == False]
    train = train[train["POLYLINE"].map(len) > 1]
    train = train[train["POLYLINE"] != "[]"]
    train = train[["POLYLINE"]]
    # Choose 10000 rows randomly from dataset to run
    train_1 = train.sample(10000)

    transformed = transform(train_1, 256)
    # Save to CSV
    transformed.to_csv("/Users/xinchengzhu/Downloads/train_transformed.csv")
