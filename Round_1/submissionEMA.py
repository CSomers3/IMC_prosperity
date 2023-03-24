from __future__ import annotations

import statistics

from datamodel import Order, TradingState, Symbol, OrderDepth

### HYPERPARAMETERS
###
# Initialize hyperparameters
FAIR_VALUE_SHIFT_AT_CROSSOVER: dict[Symbol, int] = {
    "BANANAS": 0,
    "PEARLS": 0,
    "COCONUTS": 0,
    "PINA_COLADAS": 0
}
TIME_WHILST_USING_DEFAULT_FAIR_VALUE: int = 0
PERCENT_PUT_WHEN_MM: dict[Symbol, float] = {  # not actually a percentage, but the number of shares when at 0
    "BANANAS": 20,
    "PEARLS": 20,
    "COCONUTS": 600,
    "PINA_COLADAS": 300
}
SPREAD_TO_MM: dict[Symbol, int] = {
    "BANANAS": 5,
    "PEARLS": 5,
    "COCONUTS": 5,
    "PINA_COLADAS": 5
}
EMA_SHORT_PERIOD: dict[Symbol, int] = {
    "BANANAS": 8,
    "PEARLS": 12,
    "COCONUTS": 15,
    "PINA_COLADAS": 15
}
EMA_LONG_PERIOD: dict[Symbol, int] = {
    "BANANAS": 12,
    "PEARLS": 12,
    "COCONUTS": 50,
    "PINA_COLADAS": 50
}
###



def get_top_of_book(
        order_depth: OrderDepth,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], int]:
    """
    Returns the best bids and asks of book & the corresponding volumes and the spread
    """

    # Get the top 3 bids
    bid_prices = sorted(order_depth.buy_orders.keys(), reverse=True)[:3]
    # The term "bid" refers to the price a buyer will pay to buy a specified number of shares of a stock at any
    # given time. So we want the higher ones, to sell for the highest price possible.
    best_bids = [(price, order_depth.buy_orders[price]) for price in bid_prices]

    # Get the top 3 asks
    ask_prices = sorted(order_depth.sell_orders.keys(), reverse=False)[:3]
    # The term "ask" refers to the lowest price at which a seller will sell the stock. So we want the lower ones,
    # to buy for the lowest price possible.
    best_asks = [(price, order_depth.sell_orders[price]) for price in ask_prices]

    # Return the lists of best bids and asks
    return best_bids, best_asks, best_asks[0][0] - best_bids[0][0]


def calculate_ema(prices: list[float], period: int, default_fair_value: float) -> float:
    """
    Calculates the Exponential Moving Average (EMA) for the given prices of a single product and period
    """
    if len(prices) < TIME_WHILST_USING_DEFAULT_FAIR_VALUE:
        return default_fair_value
    elif len(prices) < period:
        return sum(prices) / len(prices)

    multiplier = 2 / (period + 1)
    ema_prev = sum(prices[-period:]) / period

    for price in prices[-period + 1:]:
        ema = (price - ema_prev) * multiplier + ema_prev
        ema_prev = ema

    return ema  # noqa, len(prices) >= period


def calculate_linear_regression(x, y):
    if len(x) == 0 or len(y) == 0:
        return None

    x_mean = statistics.mean(x)
    y_mean = statistics.mean(y)

    numerator = sum((x_i - x_mean) * (y_i - y_mean) for x_i, y_i in zip(x, y))
    denominator = sum((x_i - x_mean)**2 for x_i in x)

    if denominator == 0:
        return None

    slope = numerator / denominator
    intercept = y_mean - slope * x_mean

    y_pred = [slope * x_i + intercept for x_i in x]
    residuals = [y_i - y_pred_i for y_i, y_pred_i in zip(y, y_pred)]

    return residuals  # if residuals are >0, then y_i is bigger than anticipated



class Trader:
    def __init__(self):
        # Initialize position limit and position for each product
        self.products = [
            "PEARLS",
            "BANANAS",
            "COCONUTS",
            "PINA_COLADAS"
        ]
        self.pos_limit = {
            "PEARLS": 20,
            "BANANAS": 20,
            "COCONUTS": 600,
            "PINA_COLADAS": 300
        }
        self.pos = {}
        self.min_profit = {
            "PEARLS": 0,
            "BANANAS": 0,
            "COCONUTS": 0,
            "PINA_COLADAS": 0
        }
        self.spread = SPREAD_TO_MM
        self.fair_value: dict[Symbol, float] = {
            "PEARLS": 0,
            "BANANAS": 0,
            "COCONUTS": 0,
            "PINA_COLADAS": 0
        }  # To-do: calculate them on the CSVs provided
        self.historical_prices = {product: [] for product in self.products}

    def update_fair_value(self, product: str) -> None:
        """
        Update the fair value of the given product using EMA.
        """
        short_ema = calculate_ema(
            self.historical_prices[product], EMA_SHORT_PERIOD[product], default_fair_value=self.fair_value[product]
        )
        long_ema = calculate_ema(
            self.historical_prices[product], EMA_LONG_PERIOD[product], default_fair_value=self.fair_value[product]
        )

        self.fair_value[product] = (short_ema + long_ema) / 2

        if short_ema > long_ema:
            # Short EMA is above long EMA, so we are in a bullish trend, so we set the fair value a bit higher than the
            # fair_value because we want to buy
            self.fair_value[product] += FAIR_VALUE_SHIFT_AT_CROSSOVER[product]
        else:
            # Short EMA is below long EMA, so we are in a bearish trend, so we set the fair value a bit lower than the
            # fair_value because we want to sell
            self.fair_value[product] -= FAIR_VALUE_SHIFT_AT_CROSSOVER[product]

    def market_make(
            self,
            bids: list[tuple[int, int]],
            asks: list[tuple[int, int]],
            product: str,
            cleared_best_bid: bool,
            cleared_best_ask: bool
    ):
        """
        Based on positions, make market
        """
        bid_size = int(PERCENT_PUT_WHEN_MM[product] * (1 - self.pos[product] / self.pos_limit[product]))
        ask_size = -int(PERCENT_PUT_WHEN_MM[product] * (1 + self.pos[product] / self.pos_limit[product]))

        if cleared_best_bid:
            if len(bids) > 1:
                bid = bids[1][0] + 1
            else:
                bid = bids[0][0]
        else:
            bid = bids[0][0] + 1
        if cleared_best_ask:
            if len(asks) > 1:
                ask = asks[1][0] - 1
            else:
                ask = asks[0][0]
        else:
            ask = asks[0][0] - 1

        print(f"MAKING MARKET FOR {product} WITH BID {bid} AND ASK {ask}")

        orders = [Order(product, bid, bid_size), Order(product, ask, ask_size)]

        return orders

    def pairs_trading(self, product1: str, product2: str, threshold: float) -> tuple[Order, Order] | None:
        """
        Based on the residuals of the linear regression, return the orders to be placed.
        Return a tuple (order_product_1, order_product_2) if the residuals are significant enough.
        """
        residuals = calculate_linear_regression(self.historical_prices[product1], self.historical_prices[product2])

        if residuals is None:
            return None

        last_residual = residuals[-1]

        if last_residual > threshold:
            # Long product1 and short product2
            long_order = Order(product1, self.fair_value[product1] - self.min_profit[product1], 10)
            short_order = Order(product2, self.fair_value[product1] + self.min_profit[product2], 0)
            return long_order, short_order
        elif last_residual < -threshold:
            # Short product1 and long product1
            short_order = Order(product1, self.fair_value[product1] + self.min_profit[product1], -10)
            long_order = Order(product2, self.fair_value[product2] - self.min_profit[product2], 0)
            return short_order, long_order
        else:
            return None


    def run(self, state: TradingState) -> dict[str, list[Order]]:
        """
        Only method required. It takes all buy and sell orders for all symbols as an input,
        and outputs a list of orders to be sent
        """

        # Linebreak after each timestamp (printed on the IMC end)
        print(".")

        # Initialize the method output dict as an empty dict
        result: dict[str, list[Order]] = {product: [] for product in state.order_depths.keys()}

        # Iterate over all the available products
        product: str
        for product in state.order_depths.keys():
            # Initialize the list of Orders to be sent as an empty list
            orders: list[Order] = []

            # Track if we cleared the best bid or ask
            cleared_best_bid: bool = False
            cleared_best_ask: bool = False

            # Update the position for the current product
            self.pos[product] = state.position.get(product, 0)
            print(
                f"{product}: Volume limit {self.pos_limit[product]}; position {self.pos[product]}"
            )

            # Get the top of book for the current product
            order_depth: OrderDepth = state.order_depths[product]
            best_bids: list[tuple[int, int]]
            best_asks: list[tuple[int, int]]
            best_bids, best_asks, spread = get_top_of_book(order_depth)

            # Update the historical prices for the current product
            last_price = (best_bids[0][0] + best_asks[0][0]) / 2
            self.historical_prices[product].append(last_price)

            # Update the fair value for the current product
            self.update_fair_value(product)

            if product == "PEARLS" or product == "BANANAS":
                # We are going to iterate through the sorted lists of best asks and best bids and place orders
                # accordingly, stopping when the price is no longer favorable.

                # buy everything below fair value
                ask: int
                vol: int
                for ask, vol in best_asks:  # vol is < 0
                    if ask < self.fair_value[product] - self.min_profit[product]:
                        # then buy it, if we can
                        buy_volume = min(
                            -vol, self.pos_limit[product] - self.pos[product]
                        )
                        if buy_volume > 0:
                            self.pos[product] += buy_volume
                            print(f"{product}: Buying at ${ask} x {buy_volume}")
                            orders.append(Order(product, ask, buy_volume))
                            if buy_volume == -vol:
                                cleared_best_ask = True

                # sell everything above fair value
                bid: int
                vol: int
                for bid, vol in best_bids:  # vol is > 0
                    if bid > self.fair_value[product] + self.min_profit[product]:
                        # then sell it if we can
                        sellable_volume = max(
                            -vol, -self.pos_limit[product] - self.pos[product]
                        )
                        if sellable_volume < 0:
                            self.pos[product] += sellable_volume
                            print(f"{product}: Selling at ${bid} x {sellable_volume}")
                            orders.append(Order(product, bid, sellable_volume))
                            if sellable_volume == -vol:
                                cleared_best_bid = True

                if spread > SPREAD_TO_MM[product]:
                    # We have a spread, so we need to adjust the fair value by MarketMaking that spread
                    mm = self.market_make(best_bids, best_asks, product, cleared_best_ask, cleared_best_bid)
                    orders.extend(mm)

                # Add all the above orders to the result dict
                result[product] = orders

            elif product == "COCONUTS":
                pairs_trade = self.pairs_trading("COCONUTS", "PINA_COLADAS", threshold=10)
                if pairs_trade is not None:
                    coconut_order, pinada_order = pairs_trade
                    result["COCONUTS"].append(coconut_order)
                    result["PINA_COLADAS"].append(pinada_order)

        return result
