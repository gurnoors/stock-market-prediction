def calculate_best_investment(input, budget, k_stocks=5):
    """
    Returns the stocks, which will yield the best profit.

    :param input: <type 'dict'> all stocks and their predicted highest values in the given period
    :param budget: <type 'float'> investment budget
    :return: dict, e.g:
    {
        "prophet": {
            "stocks": [
                {
                    "name": "NVDA",
                    "quantity": 23,
                    "total_price": 34.45
                },...
            ]
        }
    }
    """
    budget = float(budget)

    # take top K stocks based on profit perccentage
    profit_ratios = []  # (company_name, ratio)
    for company in input:
        max_price = float(input[company]['max_price'])
        today_price = float(input[company]['today_price'])
        ratio = max_price / today_price
        profit_ratios.append((company, ratio))
    profit_ratios = sorted(profit_ratios, key=lambda x: x[1], reverse=True)
    if len(profit_ratios) > k_stocks:
        profit_ratios = profit_ratios[:k_stocks]

    # Determine quantities of each stock in k stocks
    # Get k stocks, in quantities proportional to their profit percentages
    sum = 0
    for tuple in profit_ratios:
        sum += tuple[1]
    to_return = {
        "prophet": {
            "stocks": []
        }
    }
    total_forecasted_profit = 0
    total_price = 0
    for name, ratio in profit_ratios:
        stock = {}
        stock['name'] = name
        stock['total_price'] = (ratio / sum) * budget
        stock['quantity'] = float(stock['total_price']) / float(input[name]['today_price'])

        total_price += stock['total_price']
        gain = float(input[name]['max_price']) * stock['quantity'] - stock['total_price']
        total_forecasted_profit += gain
        stock['more_info'] = input[name]

        to_return['prophet']['stocks'].append(stock)
    to_return['prophet']['total'] = total_price
    to_return['prophet']['total_forecasted_profit'] = total_forecasted_profit
    return to_return


if __name__ == '__main__':
    sample_inp = {
        "Abb India Limited EOD Prices": {
            "max_price": "1513.58",
            "max_price_date": "2018-01-13",
            "name": "Abb India Limited EOD Prices",
            "predicted_price": "1484.982909",
            "today_price": "1509.755733"
        },
        "Aricent Ltd": {
            "max_price": "113.58",
            "max_price_date": "2018-01-13",
            "name": "Aricent Ltd",
            "predicted_price": "1484.982909",
            "today_price": "13.755733"
        }
    }
    budget = 100.11
    resp = calculate_best_investment(sample_inp, budget=budget)
    print resp
    sample_resp = {
        "prophet": {
            "total": 100.11,
            "total_forecasted_profit": 647.8597632991394,
            "stocks": [
                {
                    "more_info": {
                        "predicted_price": "1484.982909",
                        "max_price_date": "2018-01-13",
                        "today_price": "13.755733",
                        "max_price": "113.58",
                        "name": "Aricent Ltd"
                    },
                    "total_price": 89.27096106104143,
                    "name": "Aricent Ltd",
                    "quantity": 6.489727669259169
                },
                {
                    "more_info": {
                        "predicted_price": "1484.982909",
                        "max_price_date": "2018-01-13",
                        "today_price": "1509.755733",
                        "max_price": "1513.58",
                        "name": "Abb India Limited EOD Prices"
                    },
                    "total_price": 10.839038938958565,
                    "name": "Abb India Limited EOD Prices",
                    "quantity": 0.007179332856329392
                }
            ]
        }
    }
    # assert resp == sample_resp
