import pandas as pd
import pickle

def get_predictions(user_id, n=5):
    user_articles = set(clicks_df[clicks_df["user_id"]==user_id]["click_article_id"].to_list())
    scores = [(article, model.predict(user_id, article).est) for article in left_articles if article not in user_articles]
    scores.sort(key=lambda x: x[1], reverse=True)
    return [x[0] for x in scores[:n]]

clicks_df = pd.read_csv('./clicks_df.csv')
left_articles = set(clicks_df['click_article_id'].to_list())
with open(r"baseline.pickle", "rb") as input_file:
    model = pickle.load(input_file)

print(get_predictions(39))
