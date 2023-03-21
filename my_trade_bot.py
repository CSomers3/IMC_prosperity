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




class Trader:

    def __init__(self) -> None:
        self.seashells: int = 0

    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        """
        Only method required. It takes all buy and sell orders for all symbols as an input,
        and outputs a list of orders to be sent
        """
        # Initialize the method output dict as an empty dict
        result = {}

        # Iterate over all the keys (the available products) contained in the order dephts
        for product in state.order_depths.keys():

            # Check if the current product is the 'PEARLS' product, only then run the order logic
            if product == 'PEARLS' and len(state.order_depths[product].sell_orders) > 0 and len(state.order_depths[product].buy_orders) > 0:

                # Retrieve the Order Depth containing all the market BUY and SELL orders for PEARLS
                order_depth: OrderDepth = state.order_depths[product]

                # Initialize the list of Orders to be sent as an empty list
                orders: list[Order] = []

                # Define a fair value for the PEARLS.
                # Note that this value of 1 is just a dummy value, you should likely change it!
                best_bid = max(order_depth.buy_orders) 
                best_ask = min(order_depth.sell_orders)
                fair_price = min((best_bid + best_ask) / 2, 0)

                if state.position.get(product, 0) > 0:
                    bid_acceptable_price = fair_price - 4 if fair_price - 4 >= 0 else 0
                    ask_acceptable_price = fair_price + 2
                elif state.position.get(product, 0) < 0:
                    bid_acceptable_price = fair_price - 2 if fair_price - 2 >= 0 else 0
                    ask_acceptable_price = fair_price + 4
                elif state.position.get(product, 0) == 0:
                    bid_acceptable_price = fair_price - 2 if fair_price - 2 >= 0 else 0
                    ask_acceptable_price = fair_price + 2


                available_bid_lots = 20 - state.position.get(product, 0)
                available_ask_lots = 20 + state.position.get(product, 0)

                logger.print("BUY order! Price: ", bid_acceptable_price, "Volume: ", available_bid_lots)
                orders.append(Order(product, bid_acceptable_price, available_bid_lots))

                logger.print("SELL order! Price: ", ask_acceptable_price, "Volume: ", -available_ask_lots)
                orders.append(Order(product, ask_acceptable_price, -available_ask_lots))

                # Add all the above the orders to the result dict
                result[product] = orders

                # Return the dict of orders
                # These possibly contain buy or sell orders for PEARLS
                # Depending on the logic above
        logger.flush(state, result)
        return result