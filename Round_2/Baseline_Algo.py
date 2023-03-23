from __future__ import annotations
import json
from datamodel import OrderDepth, TradingState, Order, ProsperityEncoder, Symbol
from typing import Any, Dict, List


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


def market_make(self, bids, asks, product):
    """
    Based on positions, make market
    """
    bid_size = int(10 * (1 - self.pos[product] / 20))
    ask_size = -int(10 * (1 + self.pos[product] / 20))

    bid = bids[0][0] + 1
    ask = asks[0][0] - 1

    print(f"MAKING MARKET FOR {product} WITH BID {bid} AND ASK {ask}")

    orders = [Order(product, bid, bid_size), Order(product, ask, ask_size)]

    return orders


def track_bid_ask(self, product, new_bid_price, new_ask_price):
    """
    Updates the best bid/ask prices and the moving averages
    """
    if product == "BANANAS":
        # Add new prices to the lists
        self.best_bid_prices[product].append(new_bid_price)
        self.best_ask_prices[product].append(new_ask_price)

        # Limit the length of the lists to the window size
        max_len = min(len(self.best_bid_prices[product]), 10)
        self.best_bid_prices[product] = self.best_bid_prices[product][-max_len:]
        self.best_ask_prices[product] = self.best_ask_prices[product][-max_len:]

        # Calculate the moving averages
        self.bid_ma = sum(self.best_bid_prices[product]) / max_len if max_len > 0 else None
        self.ask_ma = sum(self.best_ask_prices[product]) / max_len if max_len > 0 else None
    else:
        return


def mean_revert(self, best_bids, best_asks, product, orders, fair_buy, fair_sell):
    """
    Mean-revert on the spread
    """

    if len(best_asks) > 0:
        # buy everything below our price
        for ask, vol in best_asks:
            if ask < fair_buy - self.min_profit[product]:
                order_size = min(-vol, self.pos_limit[product] - self.pos[product])
                if order_size > 0:
                    self.pos[product] += order_size
                    print(f"{product.upper()}: Buying at ${ask} x {order_size}")
                    orders.append(Order(product, ask, order_size))

    if len(best_bids) > 0:
        # sell everything above our price
        for bid, vol in best_bids:
            if bid > fair_sell + self.min_profit[product]:
                order_size = min(vol, self.pos_limit[product] - self.pos[product])
                if order_size > 0:
                    self.pos[product] -= order_size
                    print(f"{product.upper()}: Selling at ${bid} x {order_size}")
                    orders.append(Order(product, bid, -order_size))

    return orders


class Trader:
    def __init__(self):
        # Initialize position limit and position for each product
        self.products = ["PEARLS", "BANANAS"]
        self.pos_limit = {product: 20 for product in self.products}
        self.pos = {}
        self.spread = 5  # from historical data
        self.fair_value = {"PEARLS": 10000, "BANANAS": 4938.30}  # To-do: calculate them on the CSVs provided
        self.min_profit = {"PEARLS": 0.01, "BANANAS": 2.5}
        self.best_bid_prices = {"PEARLS": [], "BANANAS": []}
        self.best_ask_prices = {"PEARLS": [], "BANANAS": []}
        self.bid_ma = 0
        self.ask_ma = 0

    def run(self, state: TradingState) -> dict[str, list[Order]]:
        """
        Only method required. It takes all buy and sell orders for all symbols as an input,
        and outputs a list of orders to be sent
        """

        # Linebreak after each timestamp (printed on the IMC end)
        logger.print(".")

        # Initialize the method output dict as an empty dict
        result = {}

        # Iterate over all the available products
        product: str
        for product in state.order_depths.keys():
            if product in self.products:
                # Initialize the list of Orders to be sent as an empty list
                orders: list[Order] = []

                # Update the position for the current product
                self.pos[product] = state.position.get(product, 0)
                logger.print(f"{product.upper()}: Volume limit {self.pos_limit[product]}; position {self.pos[product]}")

                # Get the top of book for the current product
                order_depth: OrderDepth = state.order_depths[product]
                best_bids: list[tuple[int, int]]
                best_asks: list[tuple[int, int]]
                best_bids, best_asks, spread = get_top_of_book(order_depth)

                # Update the historical prices for the current product
                last_price = (best_bids[0][0] + best_asks[0][0]) / 2

                if product == "PEARLS":
                    # Check if we have buy/sell signal
                    fair_buy = fair_sell = self.fair_value[product]
                    orders = mean_revert(self, best_bids, best_asks, product, orders, fair_buy, fair_sell)

                    if spread > self.spread:
                        # We have a spread, so we need to adjust the fair value by market making that spread
                        mm = market_make(self, best_bids, best_asks, product)
                        orders.extend(mm)

                if product == "BANANAS":
                    # update the moving averages
                    track_bid_ask(self, product, best_bids[0][0], best_asks[0][0])

                    # Check if we have buy/sell signal
                    if spread < self.spread:
                        orders = mean_revert(self, best_bids, best_asks, product,
                                             orders, self.ask_ma, self.bid_ma)

                    if spread >= self.spread:
                        # We have a spread, so we need to adjust the fair value by market making that spread
                        mm = market_make(self, best_bids, best_asks, product)
                        orders.extend(mm)

                # Add all the above orders to the result dict
                result[product] = orders

        logger.print("\n")

        return result
