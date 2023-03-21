from __future__ import annotations
import json
from datamodel import Order, ProsperityEncoder, TradingState, Symbol, OrderDepth


### To be removed for the actual submission
###
class ProfitsAndLossesEstimator:
    def __init__(self, products: list[str]) -> None:
        self.profits_and_losses: dict[Symbol, int] = {product: 0 for product in products}

    def update(self, order: Order) -> None:
        """
        Updates the P&L for the given order.
        """
        self.profits_and_losses[order.symbol] = (
                self.profits_and_losses[order.symbol]  # current P&L for this symbol
                +
                order.price * (-order.quantity)  # if we buy we lose money, if we sell we gain money
        )

    def get(self, symbol: Symbol) -> int:
        """
        Returns the current P&L for the given symbol.
        """
        return self.profits_and_losses.get(symbol, 0)

    def get_all(self) -> dict[Symbol, int]:
        """
        Returns the current P&L for all symbols.
        """
        return self.profits_and_losses


class Logger:
    def __init__(self) -> None:
        self.logs = ""

    def print(self, *objects: any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]]) -> None:
        logs = self.logs
        if logs.endswith("\n"):
            logs = logs[:-1]

        print(
            json.dumps(
                {
                    "state": state,
                    "orders": orders,
                    "logs": logs,
                },
                cls=ProsperityEncoder,
                separators=(",", ":"),
                sort_keys=True,
            )
        )

        self.state = None
        self.orders = {}
        self.logs = ""


logger = Logger()
###


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
            best_bid: int | None
            best_bid_volume: int
            best_ask: int | None
            best_ask_volume: int
            best_bid, best_bid_volume, best_ask, best_ask_volume = get_top_of_book(
                order_depth
            )

            logger.print(
                f"{product.upper()}: Volume limit {self.pos_limit[product]}; position {self.pos[product]}"
            )

            # Determine if a buy order should be placed
            if best_ask and best_ask < self.fair_value[product]:
                if self.pos_limit[product] - self.pos[product] > 0:
                    buy_volume = min(
                        -best_ask_volume, self.pos_limit[product] - self.pos[product]
                    )
                    ### To be removed for the actual submission
                    ###
                    logger.print(
                        f"{product.upper()}: Buying at ${best_ask} x {buy_volume}"
                    )
                    ###
                    orders.append(Order(product, best_ask, buy_volume))
                    ### To be removed for the actual submission
                    ###
                    self.profits_and_losses_estimator.update(Order(product, best_ask, buy_volume))
                    ###

            # Determine if a sell order should be placed
            if best_bid and best_bid > self.fair_value[product]:
                if self.pos_limit[product] + self.pos[product] > 0:
                    sellable_volume = max(
                        -best_bid_volume, -self.pos_limit[product] - self.pos[product]
                    )
                    ### To be removed for the actual submission
                    ###
                    logger.print(
                        f"{product.upper()}: SELLING at ${best_bid} x {sellable_volume}"
                    )
                    ###
                    orders.append(Order(product, best_bid, sellable_volume))
                    ### To be removed for the actual submission
                    ###
                    self.profits_and_losses_estimator.update(Order(product, best_bid, sellable_volume))
                    ###

            # Add all the above orders to the result dict
            result[product] = orders


        ### To be removed for the actual submission
        ###
        # Update the P&L estimator by liquidating the position at the end of the day
        all_profits_and_losses: dict[Symbol, int] = self.profits_and_losses_estimator.get_all().copy()
        if self.timestamp <= 99900:
            for product in self.products:
                # get last mid price
                order_depth: OrderDepth = state.order_depths[product]
                best_bid, best_bid_volume, best_ask, best_ask_volume = get_top_of_book(
                    order_depth
                )
                mid_price = (best_bid + best_ask) / 2
                # simulate liquidation
                all_profits_and_losses[product] += self.pos[product] * mid_price
                logger.print(
                    f"SEASHELLS AFTER LIQUIDATION PRODUCT {product} {all_profits_and_losses[product]}"
                )
        else:
            raise Exception(
                "Problem with the self.timestamp incrementation (assuming online sandbox submission)"
            )
        ###

        logger.flush(state, result)

        self.timestamp += 100

        return result
