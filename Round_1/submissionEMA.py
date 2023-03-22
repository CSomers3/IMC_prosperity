from __future__ import annotations
from datamodel import Order, TradingState, Symbol, OrderDepth

### HYPERPARAMETERS
###
# Initialize hyperparameters
FAIR_VALUE_SHIFT_AT_CROSSOVER: dict[Symbol, int] = {
    "BANANAS": 0,
    "PEARLS": 0,
}
TIME_WHILST_USING_DEFAULT_FAIR_VALUE: int = 0
SPREAD_ADJUSTMENT: dict[Symbol, float] = {
    "BANANAS": 0,
    "PEARLS": 0,
}


###


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


class Trader:
    def __init__(self):
        # Initialize position limit and position for each product
        self.products = ["PEARLS", "BANANAS"]
        self.profits_and_losses_estimator: ProfitsAndLossesEstimator = (
            ProfitsAndLossesEstimator(self.products)
        )
        self.pos_limit = {product: 20 for product in self.products}
        self.pos = {}
        self.min_profit = 0
        self.spread = 5
        self.fair_value: dict[Symbol, float] = {
            "PEARLS": 10000,
            "BANANAS": 4938.30,
        }  # To-do: calculate them on the CSVs provided

        # EMA (Exponential Moving Average) parameters
        self.ema_short_period = 8
        self.ema_long_period = 20
        self.historical_prices = {product: [] for product in self.products}

    def update_fair_value(self, product: str) -> None:
        """
        Update the fair value of the given product using EMA.
        """
        short_ema = calculate_ema(
            self.historical_prices[product], self.ema_short_period, default_fair_value=self.fair_value[product]
        )
        long_ema = calculate_ema(
            self.historical_prices[product], self.ema_long_period, default_fair_value=self.fair_value[product]
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

    def run(self, state: TradingState) -> tuple[dict[str, list[Order]], dict[str, int]]:
        """
        Only method required. It takes all buy and sell orders for all symbols as an input,
        and outputs a list of orders to be sent
        """

        # Linebreak after each timestamp (printed on the IMC end)
        # print(".")

        # Initialize the method output dict as an empty dict
        result = {}

        # Iterate over all the available products
        product: str
        for product in state.order_depths.keys():
            # Initialize the list of Orders to be sent as an empty list
            orders: list[Order] = []

            # Update the position for the current product
            self.pos[product] = state.position.get(product, 0)
            # print(
            #     f"{product.upper()}: Volume limit {self.pos_limit[product]}; position {self.pos[product]}"
            # )

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

            if product == "PEARLS":
                if len(best_asks) > 0:
                    # buy everything below our price
                    for ask, vol in best_asks:
                        if ask < self.fair_value[product] - self.min_profit:
                            order_size = min(-vol, self.pos_limit[product] - self.pos[product])
                            if order_size > 0:
                                self.pos[product] += order_size
                                print(f"{product.upper()}: Buying at ${ask} x {order_size}")
                                orders.append(Order(product, ask, order_size))
                if len(best_bids) > 0:
                    # sell everything above our price
                    for bid, vol in best_bids:
                        if bid > (self.fair_value[product] + self.min_profit):
                            order_size = min(vol, self.pos_limit[product] - self.pos[product])
                            if order_size > 0:
                                self.pos[product] -= order_size
                                print(f"{product.upper()}: Selling at ${bid} x {order_size}")
                                orders.append(Order(product, bid, -order_size))

                if spread > self.spread:
                    # We have a spread, so we need to adjust the fair value by market making that spread
                    mm = market_make(self, best_bids, best_asks, product)
                    orders.extend(mm)

            if product == "BANANAS":
                # Adjusted fair values
                adjusted_fair_value_buy = self.fair_value[product] - (spread * SPREAD_ADJUSTMENT[product])
                # high spread then fair value smaller so buy less
                adjusted_fair_value_sell = self.fair_value[product] + (spread * SPREAD_ADJUSTMENT[product])
                # high spread then fair value bigger so sell less

                # We are going to iterate through the sorted lists of best asks and best bids and place orders accordingly,
                # stopping when the price is no longer favorable.
                # Determine if a buy order should be placed
                ask_price: int
                ask_volume: int
                for ask_price, ask_volume in best_asks:
                    if ask_price < adjusted_fair_value_buy:
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
                            # print(
                            #     f"{product.upper()}: Buying at ${ask_price} x {buy_volume}"
                            # )
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
                    if bid_price > adjusted_fair_value_sell:
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
                            # print(
                            #     f"{product.upper()}: SELLING at ${bid_price} x {sellable_volume}"
                            # )
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
        if state.timestamp <= 99900:
            for product in self.products:
                # get last mid price
                order_depth: OrderDepth = state.order_depths[product]
                best_bid: int = max(order_depth.buy_orders.keys())
                best_ask: int = min(order_depth.sell_orders.keys())
                mid_price = (best_bid + best_ask) / 2
                # simulate liquidation
                all_profits_and_losses[product] += self.pos[product] * mid_price
                # print(
                #     f"SEASHELLS AFTER LIQUIDATION PRODUCT {product} {all_profits_and_losses[product]}"
                # )
        else:
            raise Exception(
                "Problem with the self.timestamp incrementation (assuming online sandbox submission)"
            )
        ###

        # print("\n")

        ### Variable all_profits_and_losses need to be removed from the returns for the actual submission
        return result, all_profits_and_losses
        ###
