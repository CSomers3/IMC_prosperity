from __future__ import annotations
import json
from datamodel import Order, ProsperityEncoder, TradingState, Symbol, OrderDepth


### To be removed for the actual submission
###
class ProfitsAndLossesEstimator:
    def __init__(self, products: list[str]) -> None:
        self.profits_and_losses: dict[Symbol, int] = {
            product: 0 for product in products
        }

    def update(self, order: Order) -> None:
        """
        Updates the P&L for the given order.
        """
        self.profits_and_losses[order.symbol] = self.profits_and_losses[
            order.symbol
        ] + order.price * (  # current P&L for this symbol
            -order.quantity
        )  # if we buy we lose money, if we sell we gain money

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
###


def get_top_of_book(order_depth: OrderDepth) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """
    Returns the top 3 best bids and asks of book & the corresponding volumes
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
    return best_bids, best_asks


class Trader:
    def __init__(self):
        # Initialize position limit and position for each product
        self.products = ["PEARLS", "BANANAS"]
        self.profits_and_losses_estimator: ProfitsAndLossesEstimator = (
            ProfitsAndLossesEstimator(self.products)
        )
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
        print(".")

        # Initialize the method output dict as an empty dict
        result = {}

        # Iterate over all the available products
        product: str
        for product in state.order_depths.keys():
            # Initialize the list of Orders to be sent as an empty list
            orders: list[Order] = []

            # Update the position for the current product
            self.pos[product] = state.position.get(product, 0)
            print(
                f"{product.upper()}: Volume limit {self.pos_limit[product]}; position {self.pos[product]}"
            )

            order_depth: OrderDepth = state.order_depths[product]
            best_bids: list[tuple[int, int]]
            best_asks: list[tuple[int, int]]
            best_bids, best_asks = get_top_of_book(order_depth)

            # We are going to iterate through the sorted lists of best asks and best bids and place orders accordingly,
            # stopping when the price is no longer favorable.

            # Determine if a buy order should be placed
            ask_price: int
            ask_volume: int
            for ask_price, ask_volume in best_asks:
                if ask_price < self.fair_value[product]:
                    if self.pos[product] < self.pos_limit[product]:
                        # We can still buy stuff
                        buy_volume = min(
                            -ask_volume, self.pos_limit[product] - self.pos[product]
                        )
                        # Update value of self.pos[product] to reflect the new position
                        self.pos[product] = self.pos[product] + buy_volume
                        # Place the order
                        orders.append(Order(product, ask_price, buy_volume))

                        ## To be removed for the actual submission
                        ##
                        print(
                            f"{product.upper()}: Buying at ${ask_price} x {buy_volume}"
                        )
                        self.profits_and_losses_estimator.update(
                            Order(product, ask_price, buy_volume)
                        )
                        ##
                else:
                    break

            # Determine if a sell order should be placed
            bid_price: int
            bid_volume: int
            for bid_price, bid_volume in best_bids:
                if bid_price > self.fair_value[product]:
                    if self.pos[product] > -self.pos_limit[product]:
                        # We can still sell stuff
                        sellable_volume = max(
                            -bid_volume, -self.pos_limit[product] - self.pos[product]
                        )
                        # Update value of self.pos[product] to reflect the new position
                        self.pos[product] = self.pos[product] + sellable_volume
                        # Place the order
                        orders.append(Order(product, bid_price, sellable_volume))

                        ## To be removed for the actual submission
                        ##
                        print(
                            f"{product.upper()}: SELLING at ${bid_price} x {sellable_volume}"
                        )
                        self.profits_and_losses_estimator.update(
                            Order(product, bid_price, sellable_volume)
                        )
                        ##
                else:
                    break

            # Add all the above orders to the result dict
            result[product] = orders

        ### To be removed for the actual submission
        ###
        # Update the P&L estimator by liquidating the position at the end of the timestamp
        all_profits_and_losses: dict[
            Symbol, int
        ] = self.profits_and_losses_estimator.get_all().copy()
        if self.timestamp <= 99900:
            for product in self.products:
                # get last mid price
                order_depth: OrderDepth = state.order_depths[product]
                best_bid: int = max(order_depth.buy_orders.keys())
                best_ask: int = min(order_depth.sell_orders.keys())
                mid_price = (best_bid + best_ask) / 2
                # simulate liquidation
                all_profits_and_losses[product] += self.pos[product] * mid_price
                print(
                    f"SEASHELLS AFTER LIQUIDATION PRODUCT {product} {all_profits_and_losses[product]}"
                )
        else:
            raise Exception(
                "Problem with the self.timestamp incrementation (assuming online sandbox submission)"
            )
        ###

        self.timestamp += 100

        print("\n")

        return result
