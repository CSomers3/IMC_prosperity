from typing import Dict, List
from datamodel import OrderDepth, TradingState, Order


def get_top_of_book(order_depth):
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
        self.pos_limit = {product: 20 for product in self.products}
        self.pos = {}
        self.fair_value = {"PEARLS": 9999.99, "BANANAS": 4938.30}

    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        """
        Only method required. It takes all buy and sell orders for all symbols as an input,
        and outputs a list of orders to be sent
        """
        # Initialize the method output dict as an empty dict
        result = {}

        # Iterate over all the available products
        for product in state.order_depths.keys():
            # Initialize the list of Orders to be sent as an empty list
            orders: list[Order] = []

            self.pos[product] = state.position.get(product, 0)

            order_depth: OrderDepth = state.order_depths[product]

            best_bid, best_bid_volume, best_ask, best_ask_volume = get_top_of_book(order_depth)

            print(f"VOLUME LIMIT {self.pos_limit[product]} POSITION {self.pos[product]}")

            # Determine if a buy order should be placed
            if best_ask and best_ask < self.fair_value[product]:
                if self.pos_limit[product] - self.pos[product] > 0:
                    buy_volume = min(-best_ask_volume, self.pos_limit[product] - self.pos[product])
                    print(f"BUYING {product} at ${best_ask} x {buy_volume}")
                    orders.append(Order(product, best_ask, buy_volume))

            # Determine if a sell order should be placed
            if best_bid and best_bid > self.fair_value[product]:
                if self.pos_limit[product] + self.pos[product] > 0:
                    sellable_volume = max(-best_bid_volume, -self.pos_limit[product] - self.pos[product])
                    print(f"SELLING {product} at ${best_bid} x {sellable_volume}")
                    orders.append(Order(product, best_bid, sellable_volume))

            # Add all the above orders to the result dict
            result[product] = orders

        return result
