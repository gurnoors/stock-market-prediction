import json
def calculate_best_investment(input, budget):
    # ========================================
    # ({'Abb India Limited EOD Prices': {'max_price_date': '2017-12-09', 'today_price': '1509.755733', 'max_price': '1508.42', 'name': 'Abb India Limited EOD Prices', 'predicted_price': '1508.423669'}},)
    # 'budget': u'100'}
    # ========================================
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