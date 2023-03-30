from __future__ import annotations

import math
import statistics

from datamodel import Order, TradingState, Symbol, OrderDepth, Trade

### HYPERPARAMETERS
###
# Initialize hyperparameters (TO-DO REMOVE THOSE WE DON'T USE AND DON'T UPDATE FAIR VALUE IF WE DON'T USE IT)
FAIR_VALUE_SHIFT_AT_CROSSOVER: dict[Symbol, int] = {
    "BANANAS": 0,
    "PEARLS": 0,
    "COCONUTS": 0,
    "PINA_COLADAS": 0,
    "BERRIES": 0,
    "BAGUETTE": 0,
    "DIP": 0,
    "UKULELE": 0,
    "PICNIC_BASKET": 0,
}
OLIVIA_INCREASE: dict[Symbol, int] = {
    "BERRIES": 10,
    "BANANAS": 7,
    "UKULELE": 15,
}
FAIR_VALUE_ADJUSTMENT: dict[Symbol, int] = {
    "BERRIES": 3,
    "BANANAS": 1,
}
SPREAD_TO_MM: dict[Symbol, int] = {
    "BANANAS": 4,
    "PEARLS": 4,
    "BERRIES": 5,
}
EMA_SHORT_PERIOD: dict[Symbol, int] = {
    "BANANAS": 10,
    "PEARLS": 8,
    "COCONUTS": 15,
    "PINA_COLADAS": 15,
    "BERRIES": 15,
    "BAGUETTE": 15,
    "DIP": 15,
    "UKULELE": 15,
    "PICNIC_BASKET": 15,
}
EMA_LONG_PERIOD: dict[Symbol, int] = {
    "BANANAS": 12,
    "PEARLS": 1000,
    "COCONUTS": 50,
    "PINA_COLADAS": 50,
    "BERRIES": 50,
    "BAGUETTE": 100,
    "DIP": 100,
    "UKULELE": 100,
    "PICNIC_BASKET": 100,
}
MIN_PROFIT: dict[Symbol, int] = {
    "BANANAS": 0,
    "PEARLS": 0,
    "COCONUTS": 2,
    "PINA_COLADAS": 5,
    "BERRIES": 1,
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

    # Return the lists of best bids and asks and spread
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
    for i in range(1, min(11, len(x))+1):
        y_pred = slope * x[-i] + intercept
        residuals.append(y[-i] - y_pred)

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
            "BAGUETTE",
            "DIP",
            "UKULELE",
            "PICNIC_BASKET",
        ]
        self.pos_limit: dict[str, list[int]] = {
            "PEARLS": [-20, 20],
            "BANANAS": [-20, 20],
            "COCONUTS": [-600, 600],
            "PINA_COLADAS": [-300, 300],
            "BERRIES": [-250, 250],
            "DIVING_GEAR": [-50, 50],
            "BAGUETTE": [-150, 150],
            "DIP": [-300, 300],
            "UKULELE": [-70, 70],
            "PICNIC_BASKET": [-70, 70]
        }
        self.pos = {}

        self.fair_value: dict[Symbol, float] = {product: 0 for product in self.products}
        self.historical_prices = {product: [] for product in self.products}

        self.lowest_point_berries_attained = False  # lowest point hasn't been reached yet
        self.waiting_for_olivia = True

        self.need_to_buy_back = False  # for diving gear
        self.last_price_sold = 0  # for diving gear
        self.need_to_sell_back = False  # for diving gear
        self.last_price_bought = 0  # for diving gear

        # to get all the bids/asks/spreads of ETF products and not be offset by a day
        self.current_bids = {product: [] for product in ["DIP", "BAGUETTE", "UKULELE", "PICNIC_BASKET"]}
        self.current_asks = {product: [] for product in ["DIP", "BAGUETTE", "UKULELE", "PICNIC_BASKET"]}
        self.current_spread = {product: 0 for product in ["DIP", "BAGUETTE", "UKULELE", "PICNIC_BASKET"]}

        # store history of gaps between ETF and E(ETF)
        self.etf: list[float] = [
            398.64075,  # day 1
            403.26115,  # day 2
            325.27235,  # day 3
        ]

    def update_fair_value(self, product: str, timestamp: int, market_trades: dict[Symbol, list[Trade]]) -> None:
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

        if product in ["BERRIES", "BANANAS", "UKULELE"]:
            trade: Trade
            if product in market_trades:
                for trade in market_trades[product]:
                    if trade.buyer == "Olivia":
                        # if Olivia buys, then it's the lowest price of all day, so we increase the fair_value to make
                        # sure we buy a good amount
                        self.fair_value[product] += OLIVIA_INCREASE[product]
                        self.lowest_point_berries_attained = True
                        if timestamp > 750_050:
                            self.waiting_for_olivia = False
                    elif trade.seller == "Olivia":
                        # if Olivia sells, then it's the highest price of all day, so we decrease the fair_value to make
                        # sure we sell a good amount
                        self.fair_value[product] -= OLIVIA_INCREASE[product]

        if product == "BERRIES":
            if 400_000 <= timestamp <= 557380:
                self.fair_value[product] += FAIR_VALUE_ADJUSTMENT[product]  # fair_value is too low, we know that
                # the price will go up, so we increase the fair_value to make sure we buy
            if timestamp >= 600_000:
                self.fair_value[product] -= FAIR_VALUE_ADJUSTMENT[product]  # fair_value is too high, we know that
                # the price will go down, so we decrease the fair_value to make sure we sell

        if product == "BANANA":
            self.fair_value[product] -= FAIR_VALUE_ADJUSTMENT[product]  # usually bananas price decreases, so we
            # decrease the fair_value to make sure we sell more than we buy

        # if short_ema > long_ema:
        #     # Short EMA is above long EMA, so we are in a bullish trend, so we set the fair value a bit higher than
        #     # the fair_value because we want to buy
        #     self.fair_value[product] += FAIR_VALUE_SHIFT_AT_CROSSOVER[product]
        # else:
        #     # Short EMA is below long EMA, so we are in a bearish trend, so we set the fair value a bit lower than
        #     # the fair_value because we want to sell
        #     self.fair_value[product] -= FAIR_VALUE_SHIFT_AT_CROSSOVER[product]

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
        bid_size = self.pos_limit[product][1] - self.pos[product]
        ask_size = self.pos_limit[product][0] - self.pos[product]

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
            if (
                    self.pos[product1] + 10 > self.pos_limit[product1][1]
                    or
                    self.pos[product2] - 10 < self.pos_limit[product2][0]
            ):
                return None
            else:
                long_order = Order(product1, int(self.fair_value[product1] - MIN_PROFIT[product1]), 10)
                short_order = Order(product2, math.ceil(self.fair_value[product2] + MIN_PROFIT[product2]), -10)
                return long_order, short_order
        elif average_residual < -threshold:  # if residuals are <0, then product 2 is cheaper than anticipated by
            # product 1 short product1 and long product1
            if (
                    self.pos[product1] - 10 < self.pos_limit[product1][0]
                    or
                    self.pos[product2] + 10 > self.pos_limit[product2][1]
            ):
                return None
            else:
                short_order = Order(product1, math.ceil(self.fair_value[product1] + MIN_PROFIT[product1]), -10)
                long_order = Order(product2, int(self.fair_value[product2] - MIN_PROFIT[product2]), 10)
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

        # Initialize first_time_coco_pina
        first_time_coco_pina = True

        # Iterate over all the available products to update all the positions
        product: str
        for product in state.order_depths.keys():
            # Update the position for the current product
            self.pos[product] = state.position.get(product, 0)
            print(
                f"{product}: Volume limit {self.pos_limit[product]}; position {self.pos[product]}"
            )

        # Iterate over the ETF products to update self.current_bids, self.current_spread, etc ...
        for product in ["DIP", "BAGUETTE", "UKULELE", "PICNIC_BASKET"]:
            # Get the top of book for the current product
            order_depth: OrderDepth = state.order_depths[product]
            best_bids: list[tuple[int, int]]
            best_asks: list[tuple[int, int]]
            best_bids, best_asks, spread = get_top_of_book(order_depth)

            # Update the current bids, asks and spread for the current product
            self.current_bids[product] = best_bids
            self.current_asks[product] = best_asks
            self.current_spread[product] = spread

            # Update the historical prices for the current product
            last_price = (best_bids[0][0] + best_asks[0][0]) / 2
            self.historical_prices[product].append(last_price)

        # Iterate over all the available products to place the orders
        for product in state.order_depths.keys():
            # Initialize the list of Orders to be sent as an empty list
            orders: list[Order] = []

            # Track if we cleared the best bid or ask (for the MM)
            cleared_best_bid: bool = False
            cleared_best_ask: bool = False

            # Get the top of book for the current product
            if product not in ["DIP", "BAGUETTE", "UKULELE", "PICNIC_BASKET"]:
                # done separetely for those (cf. above)
                order_depth: OrderDepth = state.order_depths[product]
                best_bids: list[tuple[int, int]]
                best_asks: list[tuple[int, int]]
                best_bids, best_asks, spread = get_top_of_book(order_depth)

                # Update the historical prices for the current product
                last_price = (best_bids[0][0] + best_asks[0][0]) / 2
                self.historical_prices[product].append(last_price)

            # Update the fair value for the current product, but only for the products we need to have access to the
            # fair value, otherwise we compute stuff for nothing
            if product in [
                "PEARLS", "BANANAS",
                "BERRIES",
                "COCONUTS", "PINA_COLADAS",
                "DIP", "BAGUETTE", "UKULELE", "PICNIC_BASKET"
            ]:
                self.update_fair_value(product, state.timestamp, state.market_trades)

            if product in ["PEARLS", "BANANAS", "BERRIES"]:
                if product == "BERRIES" and 750_010 <= state.timestamp <= 750_050:
                    # try to get back to a neutral position or to -250 if lowest_point_berries_attained is False.
                    if self.lowest_point_berries_attained:
                        # then we try to get to neutral
                        orders.append(
                            Order(product, best_asks[0][0], -self.pos[product])
                        )
                    else:
                        # then we try to get to -250 because lowest point (at which we will buy back everything)
                        # has not been attained yet
                        orders.append(
                            Order(product, best_asks[0][0], -self.pos[product] + self.pos_limit[product][0])
                        )
                        self.waiting_for_olivia = True  # we are now not trading, we just wait for Olivia to buy back
                        # as a signal that the lowest point has been attained
                else:
                    # We are going to iterate through the sorted lists of best asks and best bids and place orders
                    # accordingly, stopping when the price is no longer favorable.

                    if product == "BERRIES" and state.timestamp > 750_050 and self.waiting_for_olivia:
                        pass  # we just chilling, we'll buy when lowest_point_berries_attained is True
                    else:
                        # buy everything below fair value
                        ask: int
                        vol: int
                        for ask, vol in best_asks:  # vol is < 0
                            if ask < self.fair_value[product] - MIN_PROFIT[product]:
                                # then buy it, if we can
                                buy_volume = min(
                                    -vol, self.pos_limit[product][1] - self.pos[product]
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
                                    -vol, self.pos_limit[product][0] - self.pos[product]
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

                        if state.timestamp > 750_050 and product == "BERRIES" and not self.waiting_for_olivia:
                            self.waiting_for_olivia = True  # we are no longer wanting to trade berries
                            # so we wait for a signal that will never come

                # Add all the above orders to the result dict
                result[product] = orders

            elif product in ["COCONUTS", "PINA_COLADAS"]:
                if not first_time_coco_pina:
                    pairs_trade = self.pairs_trading("COCONUTS", "PINA_COLADAS", threshold=5)
                    if pairs_trade is not None:
                        coconut_order, pinada_order = pairs_trade
                        result["COCONUTS"].append(coconut_order)
                        result["PINA_COLADAS"].append(pinada_order)
                else:
                    first_time_coco_pina = False

            elif product == "DIVING_GEAR":
                # Update Dolphin observations (price)
                dolphin_price = state.observations['DOLPHIN_SIGHTINGS']
                print(f"DOLPHINS: {dolphin_price}")
                self.historical_prices["DOLPHINS"].append(dolphin_price)

                # If big drop or big spike* in dolphin sightings in the last timestamp compared to the average last 5
                # timestamps, then buy or sell diving gear
                # * +/- 10
                if len(self.historical_prices["DOLPHINS"]) > 5:
                    mean_last_5 = sum(self.historical_prices["DOLPHINS"][-6:-1]) / 5
                    if dolphin_price > mean_last_5 + 5:
                        already_bought = 0
                        for k in range(len(best_asks)):
                            # Buy as much diving gear as possible
                            orders.append(Order(
                                "DIVING_GEAR",
                                best_asks[k][0],
                                self.pos_limit["DIVING_GEAR"][1]-self.pos["DIVING_GEAR"]-already_bought)
                            )
                            already_bought -= best_asks[k][1]
                            print(
                                f"DIVING_GEAR: "
                                f"Buying at ${best_asks[k][0]}"
                                f" x "
                                f"{self.pos_limit['DIVING_GEAR'][1]-self.pos['DIVING_GEAR']}"
                            )
                        self.need_to_sell_back = True
                        self.last_price_bought = best_asks[0][0]
                    elif dolphin_price < mean_last_5 - 5:
                        already_sold = 0
                        for k in range(len(best_bids)):
                            # Sell as much diving gear as possible
                            orders.append(Order(
                                "DIVING_GEAR",
                                best_bids[k][0],
                                self.pos_limit["DIVING_GEAR"][0]-self.pos["DIVING_GEAR"]+already_sold)
                            )
                            already_sold += best_bids[k][1]
                            print(
                                f"DIVING_GEAR: "
                                f"Selling at ${best_bids[k][0]}"
                                f" x "
                                f"{self.pos_limit['DIVING_GEAR'][0]-self.pos['DIVING_GEAR']}"
                            )
                        self.need_to_buy_back = True
                        self.last_price_sold = best_bids[0][0]

                    if self.need_to_sell_back:
                        already_sold = 0
                        for k in range(len(best_bids)):
                            if best_bids[k][0] > self.last_price_bought + 300:
                                # new price which we sell the stuff is higher than the price we bought it for
                                orders.append(Order(
                                    "DIVING_GEAR",
                                    best_bids[k][0],
                                    self.pos_limit["DIVING_GEAR"][0]-self.pos["DIVING_GEAR"]+already_sold)
                                )
                                already_sold += best_bids[k][1]
                                print(
                                    f"DIVING_GEAR: "
                                    f"Selling at ${best_bids[k][0]}"
                                    f" x "
                                    f"{self.pos_limit['DIVING_GEAR'][0]-self.pos['DIVING_GEAR']}"
                                )
                                self.need_to_sell_back = False

                    if self.need_to_buy_back:
                        already_bought = 0
                        for k in range(len(best_asks)):
                            if best_asks[k][0] < self.last_price_sold - 300:
                                # new price which we buy the stuff is lower than the price we sold it for
                                orders.append(Order(
                                    "DIVING_GEAR",
                                    best_asks[k][0],
                                    self.pos_limit["DIVING_GEAR"][1]-self.pos["DIVING_GEAR"]-already_bought)
                                )
                                already_bought -= best_asks[k][1]
                                print(
                                    f"DIVING_GEAR: "
                                    f"Buying at ${best_asks[k][0]}"
                                    f" x "
                                    f"{self.pos_limit['DIVING_GEAR'][1]-self.pos['DIVING_GEAR']}"
                                )
                                self.need_to_buy_back = False

                    result["DIVING_GEAR"] = orders

            elif product == "PICNIC_BASKET":
                # Basket = 2*Baguette + 4*Dip + 1*Ukulele
                weights = {"BAGUETTE": 2, "DIP": 4, "UKULELE": 1}
                expected_etf = (
                        self.historical_prices["BAGUETTE"][-1] * weights["BAGUETTE"] +
                        self.historical_prices["DIP"][-1] * weights["DIP"] +
                        self.historical_prices["UKULELE"][-1]
                )

                self.etf.append(self.historical_prices["PICNIC_BASKET"][-1] - expected_etf)

                if state.timestamp > 5_000:
                    mean = sum(self.etf) / len(self.etf)
                    std = (
                                  sum((x - mean) ** 2 for x in self.etf) / len(self.etf)
                    ) ** 0.5

                    threshold = std * 1.5

                    # Calculate component contributions to deviation
                    component_deviation = {}
                    for item, weight in weights.items():
                        component_deviation[item] = (
                                self.historical_prices[item][-1] * weight -
                                sum(self.historical_prices[item][-10:-1]) * weight/9
                        )
                    # Identify the component with the highest absolute contribution
                    max_deviation_component = max(component_deviation, key=lambda x: abs(component_deviation[x]))

                    if self.etf[-1] > mean + threshold:
                        # PICNIC_BASKET is overvalued, sell it
                        quantity = max(
                            -self.current_bids["PICNIC_BASKET"][0][1],
                            self.pos_limit["PICNIC_BASKET"][0] - self.pos["PICNIC_BASKET"]
                        )
                        orders.append(Order(
                            "PICNIC_BASKET",
                            self.current_bids["PICNIC_BASKET"][0][0],
                            quantity)
                        )
                        print(
                            f"PICNIC_BASKET: "
                            f"Selling at ${self.current_bids['PICNIC_BASKET'][0][0]}"
                            f" x "
                            f"{quantity}"
                        )

                        # Hedge with BAGUETTE, DIP, or UKULELE by buying them
                        for item, weight in weights.items():
                            if item == max_deviation_component:
                                buying_quantity: int = min(-weight * quantity, self.pos_limit[item][1] - self.pos[item])
                                result[item].append(Order(
                                    item,
                                    self.current_asks[item][0][0],
                                    buying_quantity
                                ))
                                print(
                                    f"{item}: "
                                    f"Buying at ${self.current_asks[item][0][0]}"
                                    f" x "
                                    f"{buying_quantity}"
                                )
                                break


                    elif self.etf[-1] < mean - threshold:
                        # PICNIC_BASKET is undervalued, buy it
                        quantity = min(
                            -self.current_asks["PICNIC_BASKET"][0][1],
                            self.pos_limit["PICNIC_BASKET"][1] - self.pos["PICNIC_BASKET"]
                        )
                        orders.append(Order(
                            "PICNIC_BASKET",
                            self.current_asks['PICNIC_BASKET'][0][0],
                            quantity)
                        )
                        print(
                            f"PICNIC_BASKET: "
                            f"Buying at ${self.current_asks['PICNIC_BASKET'][0][0]}"
                            f" x "
                            f"{quantity}"
                        )

                        # Hedge with BAGUETTE, DIP, or UKULELE by selling them
                        for item, weight in weights.items():
                            if item == max_deviation_component:
                                selling_quantity = max(-weight * quantity, self.pos_limit[item][0] - self.pos[item])
                                result[item].append(Order(
                                    item,
                                    self.current_bids[item][0][0],
                                    selling_quantity
                                ))
                                print(
                                    f"{item}: "
                                    f"Selling at ${self.current_bids[item][0][0]}"
                                    f" x "
                                    f"{selling_quantity}"
                                )
                                break

                    result["PICNIC_BASKET"] = orders

        return result
