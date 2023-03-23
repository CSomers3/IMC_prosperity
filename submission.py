#!/usr/bin/env python
# coding: utf-8

# In[4]:


from __future__ import annotations
import json
from datamodel import Order, ProsperityEncoder, TradingState, Symbol, OrderDepth


def get_top_of_book(order_depth: OrderDepth) -> tuple[int | None, int, int | None, int]:
    """
    Returns the top of book (best bid/ask) & the corresponding volumes
    """
    best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
    best_bid_volume = order_depth.buy_orders.get(best_bid, 0) if best_bid else 0
    best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
    best_ask_volume = order_depth.sell_orders.get(best_ask, 0) if best_ask else 0

    return best_bid, best_bid_volume, best_ask, best_ask_volume


class Trader:
    def __init__(self):
        # Initialize position limit and position for each product
        self.products = ["PEARLS", "BANANAS"]
        self.profits_and_losses_estimator: ProfitsAndLossesEstimator = ProfitsAndLossesEstimator(self.products)
        self.pos_limit = {product: 20 for product in self.products}
        self.pos = {}
        self.fair_value = {"PEARLS": 9999.99, "BANANAS": 4938.30}
        self.timestamp: int = 0

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
            # Initialize the list of Orders to be sent as an empty list
            orders: list[Order] = []

            # Update the position for the current product
            self.pos[product] = state.position.get(product, 0)

            order_depth: OrderDepth = state.order_depths[product]
            # Compute the rolling mean of the bid and ask prices
            window_size = 10
            bid_prices = list(order_depth.buy_orders.keys())[:window_size]
            ask_prices = list(order_depth.sell_orders.keys())[:window_size]
            bid_mean = sum(bid_prices) / len(bid_prices)
            ask_mean = sum(ask_prices) / len(ask_prices)
            
            # Determine if a buy order should be placed
            if ask_mean < self.fair_value[product]:
                if self.pos_limit[product] - self.pos[product] > 0:
                    buy_volume = min(
                        -order_depth.sell_orders.get(ask_prices[0], 0),
                        self.pos_limit[product] - self.pos[product]
                    )
                    orders.append(Order(product, ask_prices[0], buy_volume))

            # Determine if a sell order should be placed
            if bid_mean > self.fair_value[product]:
                if self.pos_limit[product] + self.pos[product] > 0:
                    sellable_volume = max(
                        -order_depth.buy_orders.get(bid_prices[0], 0),
                        -self.pos_limit[product] - self.pos[product]
                    )
                    orders.append(Order(product, bid_prices[0], sellable_volume))
                    

            # Add all the above orders to the result dict
            result[product] = orders
    
        logger.flush(state, result)

        self.timestamp += 100

        return result

