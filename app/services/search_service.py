from app.services.play_service import show_resut
import pandas as pd



def sum_data_class(Data):
    df = pd.DataFrame(Data)
    person_data = df.groupby('track_id')
    sum_data = {}
    for track,data in person_data:
        unique_class = pd.unique(data['class_id'])
        sum_data[track] = unique_class
    return sum_data


def search_by_class(data, class_id: int):
    try:
        result = {}
        for key,item in data.items():
            if class_id in item:
                result[key] = item

        return result

    except Exception as e:
        print(f'[search_by_class] is error: {e}')

def find_max_color(colors):#input list of colors
    max_color = max(colors.items(), key=lambda x: x[1])
    return max_color[0] #[color,value]


def search_by_color(data, colors: list[str] = [], collap=False):
    try:
        df = pd.DataFrame(data)
        try:
            df["max_colors"] = df["colors"].apply(lambda x:find_max_color(x))

        except:
            df["max_colors"] = df["mean_color_bgr"].apply(lambda x:find_max_color(x))

        filtered = df[df['max_colors'].isin(colors)]
        unique_track = pd.unique(filtered['track_id'])
        df = df[df['track_id'].isin(unique_track)]
        return df

        # df = df[df[column].apply(lambda x: x[0] in colors)]
    except Exception as e:
        print(f'[search_by_color] is error: {e}')


def search_process(cond:list[list[int,list[str]]], Data, cloth_collap=True):
    df = pd.DataFrame(Data)
    for item in cond:
        if cloth_collap:
            sum_data = sum_data_class(df)
            result = search_by_class(sum_data,item[0])
            track_this = list(result.keys())
            df = df[df['track_id'].isin(track_this)]
            df = search_by_color(df, item[1])

        else:
            sum_data = sum_data_class(df)
            result = search_by_class(sum_data,item[0])
            track_this = list(result.keys())
            df = df[df['track_id'].isin(track_this)]
            df = search_by_color(df, item[1])
    return df

