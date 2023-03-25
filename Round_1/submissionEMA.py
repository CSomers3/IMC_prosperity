from __future__ import annotations

import math
import statistics

from datamodel import Order, TradingState, Symbol, OrderDepth

### HYPERPARAMETERS
###
# Initialize hyperparameters
FAIR_VALUE_SHIFT_AT_CROSSOVER: dict[Symbol, int] = {
    "BANANAS": 0,
    "PEARLS": 0,
    "COCONUTS": 0,
    "PINA_COLADAS": 0,
    "BERRIES": 0,
    "DIVING_GEAR": 0
}
PERCENT_PUT_WHEN_MM: dict[Symbol, float] = {  # not actually a percentage, but the number of shares when at 0
    "BANANAS": 20,
    "PEARLS": 20,
    "COCONUTS": 600,
    "PINA_COLADAS": 300,
    "BERRIES": 250,
    "DIVING_GEAR": 50
}
SPREAD_TO_MM: dict[Symbol, int] = {
    "BANANAS": 4,
    "PEARLS": 4,
    "COCONUTS": 5,
    "PINA_COLADAS": 5,
    "BERRIES": 5,
    "DIVING_GEAR": 5
}
EMA_SHORT_PERIOD: dict[Symbol, int] = {
    "BANANAS": 10,
    "PEARLS": 8,
    "COCONUTS": 15,
    "PINA_COLADAS": 15,
    "BERRIES": 15,
    "DIVING_GEAR": 15
}
EMA_LONG_PERIOD: dict[Symbol, int] = {
    "BANANAS": 12,
    "PEARLS": 60,
    "COCONUTS": 50,
    "PINA_COLADAS": 50,
    "BERRIES": 50,
    "DIVING_GEAR": 50
}
MIN_PROFIT: dict[Symbol, int] = {
    "BANANAS": 0,
    "PEARLS": 0,
    "COCONUTS": 0,
    "PINA_COLADAS": 0,
    "BERRIES": 0,
    "DIVING_GEAR": 0
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


def calculate_ema(prices: list[float], period: int) -> float:
    """
    Calculates the Exponential Moving Average (EMA) for the given prices of a single product and period
    """
    if len(prices) < period:
        return sum(prices) / len(prices)

    multiplier = 2 / (period + 1)
    ema_prev = sum(prices[-period:]) / period

    for price in prices[-period + 1:]:
        ema = (price - ema_prev) * multiplier + ema_prev
        ema_prev = ema

    return ema  # noqa, len(prices) >= period


def calculate_linear_regression(x, y):
    """
    Calculates the linear regression of the given x and y values
    Returns the residuals of the regression for the past 10 timestamps
    """
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

    residuals = []
    for i in range(min(10, len(x), len(y))):  # x and y might have different lengths since one is updated before the
        # other one, and this function is called between the two
        y_pred = slope * x[i] + intercept
        residuals.append(y[i] - y_pred)

    return sum(residuals)/len(residuals)  # if residuals are >0, then y_i is bigger than anticipated



class Trader:
    def __init__(self):
        # Initialize position limit and position for each product
        self.products = [
            "PEARLS",
            "BANANAS",
            "COCONUTS",
            "PINA_COLADAS",
            "BERRIES",
            "DIVING_GEAR",
            "DOLPHINS",
        ]
        self.pos_limit = {
            "PEARLS": 20,
            "BANANAS": 20,
            "COCONUTS": 600,
            "PINA_COLADAS": 300,
            "BERRIES": 250,
            "DIVING_GEAR": 50
        }
        self.pos = {}
        self.spread = SPREAD_TO_MM
        self.fair_value: dict[Symbol, float] = {product: 0 for product in self.products}
        self.historical_prices = {product: [] for product in self.products}

    def update_fair_value(self, product: str) -> None:
        """
        Update the fair value of the given product using EMA.
        """
        short_ema = calculate_ema(
            self.historical_prices[product], EMA_SHORT_PERIOD[product]
        )
        long_ema = calculate_ema(
            self.historical_prices[product], EMA_LONG_PERIOD[product]
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
        Assumption: product 1 predicts product 2
        Based on the residuals of the linear regression, return the orders to be placed.
        Return a tuple (order_product_1, order_product_2) if the residuals are significant enough.
        """
        average_residual: float = calculate_linear_regression(
            self.historical_prices[product1],
            self.historical_prices[product2]
        )  # averaged over past values, cf calculate_linear_regression

        if average_residual is None:
            return None

        if average_residual > threshold:  # if residuals are >0, then product 2 is more expensive than anticipated by
            # product 1, long product1 and short product2
            long_order = Order(product1, int(self.fair_value[product1] - MIN_PROFIT[product1]), -10)
            short_order = Order(product2, math.ceil(self.fair_value[product2] + MIN_PROFIT[product2]), 10)
            return long_order, short_order
        elif average_residual < -threshold:  # if residuals are <0, then product 2 is cheaper than anticipated by
            # product 1 short product1 and long product1
            short_order = Order(product1, math.ceil(self.fair_value[product1] + MIN_PROFIT[product1]), 10)
            long_order = Order(product2, int(self.fair_value[product2] - MIN_PROFIT[product2]), -10)
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

            if product in ["PEARLS", "BANANAS", "BERRIES"]:
                # We are going to iterate through the sorted lists of best asks and best bids and place orders
                # accordingly, stopping when the price is no longer favorable.

                # buy everything below fair value
                ask: int
                vol: int
                for ask, vol in best_asks:  # vol is < 0
                    if ask < self.fair_value[product] - MIN_PROFIT[product]:
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
                    if bid > self.fair_value[product] + MIN_PROFIT[product]:
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
                pairs_trade = self.pairs_trading("COCONUTS", "PINA_COLADAS", threshold=5)
                if pairs_trade is not None:
                    coconut_order, pinada_order = pairs_trade
                    result["COCONUTS"].append(coconut_order)
                    result["PINA_COLADAS"].append(pinada_order)

            elif product == "DIVING_GEAR":
                # Update Dolphin observations (price)
                dolphin_price = state.observations['DOLPHIN_SIGHTINGS']
                print(f"DOLPHINS: {dolphin_price}")
                self.historical_prices["DOLPHINS"].append(dolphin_price)

                residuals: float = calculate_linear_regression(
                    self.historical_prices["DOLPHINS"],
                    self.historical_prices["DIVING_GEAR"]
                )

                if residuals is not None:
                    if residuals > 0:
                        # diving gear more expensive than dolphin-derived-demand
                        print(f"Residuals: {residuals}>0, so shorting diving gear")
                        sellable_volume = max(-best_bids[0][1], -self.pos_limit[product] - self.pos[product])
                        short = Order("DIVING_GEAR",
                                      math.ceil(self.fair_value[product] + MIN_PROFIT[product]),
                                      sellable_volume)
                        result["DIVING_GEAR"].append(short)

                    elif residuals < 0:
                        # diving gear cheaper than dolphin-derived-demand
                        print(f"Residuals: {residuals}<0, so long diving gear")
                        buyable_volume = min(-best_asks[0][1], self.pos_limit[product] - self.pos[product])
                        long = Order("DIVING_GEAR", int(self.fair_value[product] - MIN_PROFIT[product]),
                                     buyable_volume)
                        result["DIVING_GEAR"].append(long)

        return result
