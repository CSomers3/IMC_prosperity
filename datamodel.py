import json
from typing import Dict, List
from json import JSONEncoder

Time = int
Symbol = str
Product = str
Position = int
UserId = str
Observation = int


class Listing:
    def __init__(self, symbol: Symbol, product: Product, denomination: Product):
        self.symbol = symbol
        self.product = product
        self.denomination = denomination


class Order:
    def __init__(self, symbol: Symbol, price: int, quantity: int) -> None:
        self.symbol = symbol
        self.price = price
        self.quantity = quantity
        """
        The quantity of the order: the maximum quantity that the algorithm wishes to buy or sell.
        If the sign of the quantity is positive, the order is a buy order, if the sign of the quantity is negative,
        it is a sell order.
        """

    def __str__(self) -> str:
        return (
            "(" + self.symbol + ", " + str(self.price) + ", " + str(self.quantity) + ")"
        )

    def __repr__(self) -> str:
        return (
            "(" + self.symbol + ", " + str(self.price) + ", " + str(self.quantity) + ")"
        )


class OrderDepth:
    """
    All the orders on a single side (buy or sell) are aggregated in a dict, where the keys indicate the price
    associated with the order, and the corresponding keys indicate the total volume on that price level.
    """
    def __init__(self):
        self.buy_orders: Dict[int, int] = {}  # values are positive
        self.sell_orders: Dict[int, int] = {}  # values are negative


class Trade:
    def __init__(
        self,
        symbol: Symbol,
        price: int,
        quantity: int,
        buyer: UserId = None,
        seller: UserId = None,
        timestamp: int = 0,
    ) -> None:
        self.symbol = symbol
        self.price: int = price
        self.quantity: int = quantity
        self.buyer = buyer
        self.seller = seller
        self.timestamp = timestamp


class TradingState(object):
    own_trades = None

    def __init__(
        self,
        timestamp: Time,
        listings: Dict[Symbol, Listing],
        order_depths: Dict[Symbol, OrderDepth],
        own_trades: Dict[Symbol, List[Trade]],
        market_trades: Dict[Symbol, List[Trade]],
        position: Dict[Product, Position],
        observations: Dict[Product, Observation],
    ):
        self.timestamp = timestamp
        self.listings = listings
        self.order_depths = order_depths
        self.own_trades = own_trades
        self.market_trades = market_trades
        self.position = position
        self.observations = observations

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True)


class ProsperityEncoder(JSONEncoder):
    def default(self, o):
        return o.__dict__
