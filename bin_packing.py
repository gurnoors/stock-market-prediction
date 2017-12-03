import json
def calculate_best_investment(*args, **kwargs):
    sample = {
  "prophet": {
    "stocks": [
      {
        "name": "NVDA",
        "quantity": 23,
        "total_price": 34.45
      },
      {
        "name": "GOOG",
        "quantity": 22,
        "total_price": 45.45
      },
      {
        "name": "MSFT",
        "quantity": 21,
        "total_price": 64.57
      }
    ],
    "total": 45
  },
  "lstm": {
    "stocks": [
      {
        "name": "NVDA",
        "quantity": 23,
        "total_price": 34.45
      },
      {
        "name": "GOOG",
        "quantity": 22,
        "total_price": 45.45
      },
      {
        "name": "MSFT",
        "quantity": 21,
        "total_price": 64.57
      }
    ],
    "total": 45,
    "_comment": "total price"

  },
  "sarima": {},
  "somethingelse": {}
}
    sample_json = json.dumps(sample)
    return sample_json


if __name__ == '__main__':
    calculate_best_investment()