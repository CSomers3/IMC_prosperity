from __future__ import annotations
import json
from datamodel import Order, ProsperityEncoder, TradingState, Symbol, OrderDepth
from typing import Any, Dict, List

class Logger:
    def __init__(self) -> None:
        self.logs = ""

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: Dict[Symbol, List[Order]]) -> None:
        logs = self.logs
        if logs.endswith("\n"):
            logs = logs[:-1]

        print(json.dumps({
            "state": state,
            "orders": orders,
            "logs": logs,
        }, cls=ProsperityEncoder, separators=(",", ":"), sort_keys=True))

        self.state = None
        self.orders = {}
        self.logs = ""

logger = Logger()


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
        self.pos_limit = {product: 20 for product in self.products}
        self.pos = {}
        self.fair_value = {"PEARLS": 9999.99, "BANANAS": 4938.30}

        self.seashells: int = 0
        """
        Relative amount of seashells made that day so far
        """

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
            best_bid: int | None
            best_bid_volume: int
            best_ask: int | None
            best_ask_volume: int
            best_bid, best_bid_volume, best_ask, best_ask_volume = get_top_of_book(order_depth)

            logger.print(f"{product.upper()}: Volume limit {self.pos_limit[product]}; position {self.pos[product]}")

            # Determine if a buy order should be placed
            if best_ask and best_ask < self.fair_value[product]:
                if self.pos_limit[product] - self.pos[product] > 0:
                    buy_volume = min(-best_ask_volume, self.pos_limit[product] - self.pos[product])
                    logger.print(f"{product.upper()}: Buying at ${best_ask} x {buy_volume}")
                    orders.append(Order(product, best_ask, buy_volume))
                    self.seashells -= best_ask * buy_volume

            # Determine if a sell order should be placed
            if best_bid and best_bid > self.fair_value[product]:
                if self.pos_limit[product] + self.pos[product] > 0:
                    sellable_volume = max(-best_bid_volume, -self.pos_limit[product] - self.pos[product])
                    logger.print(f"{product.upper()}: SELLING at ${best_bid} x {sellable_volume}")
                    orders.append(Order(product, best_bid, sellable_volume))
                    self.seashells += best_bid * (-sellable_volume)

            # Add all the above orders to the result dict
            result[product] = orders

        logger.print(f"SEASHELLS AT TIMESTAMP {self.timestamp}: {self.seashells}")

        self.timestamp += 100
        logger.flush(state, result)
        return result
