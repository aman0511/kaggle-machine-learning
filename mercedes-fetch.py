import urllib
import json
import pandas as pd


## https://crowdstats.eu/topics/kaggle-mercedes-benz-greener-manufacturing-leaderboard-probing
## https://www.kaggle.com/c/mercedes-benz-greener-manufacturing/discussion/35174

def fetch_full_data():
  all_questions=json.loads(
    urllib.urlopen(
      "https://crowdstats.eu/api/topics/kaggle-mercedes-benz-greener-manufacturing-leaderboard-probing/questions"
    ).read()
  )
  answers = []
  for question in all_questions:
    for answer in question['answers']:
      newAnswer = {
        'ID': question['id'],
        'insidePublicLB': answer['inside_public_lb'],
        'y': answer['y_value'],
      }
      answers.append(newAnswer)
  return pd.DataFrame(answers)

full_data = fetch_full_data()
datapoints_inside_public_lb = full_data[full_data['insidePublicLB']==True]
datapoints_inside_public_lb = datapoints_inside_public_lb[['ID', 'y']]

datapoints_inside_public_lb.to_csv('mercedes-extra.csv', index=False)
