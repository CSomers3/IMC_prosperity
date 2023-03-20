# IMC_prosperity - Team Achill Island
Private Github for the IMC Prosperity Challenge


# Prosperity Outline
## allowed python imports:
- pandas
- NumPy
- statistics
- math
- typing

## info:
**rounds:** 
1. 20 March, 10:00
2. 22 March, 10:00
3. 24 March, 10:00
4. 26 March, 10:00
5. 28 March, 10:00
:
- some rounds will also require **manual trading** that are only **24 hours** long!

**submissions:**
- python 3.9 scrippie
- scrippies can submit as many as we want but only one (latest?) will be active and used
- manual trading can also be resubmitted but latest will be used. 

**The challenge:**
- beginning of new round, new item, with sample data
- every new round, old product(s) will be tradable aswell (allows for optimization of algorithm)


**Simulation**
- alrgorithm will be written in ```run``` method of the ```Trader``` class
- every iteration of the simulation will execute the ```run``` method and be provided with  the ```TradingState``` object.
- ```TradingState``` contains: 
    - Contains overview of the trades of last iteration (of both alforithm and other participants)
    - Per product overview of outstanding buy/sell orders from bots

## code snippets
**Trader class**
```python
# The Python code below is the minimum code that is required in a submission file:
# 1. The "datamodel" imports at the top. Using the typing library is optional.
# 2. A class called "Trader", this class name should not be changed.
# 3. A run function that takes a tradingstate as input and outputs a "result" dict.

from typing import Dict, List
from datamodel import OrderDepth, TradingState, Order

class Trader:

    def run(self, state: TradingState) -> Dict[str, List[Order]]:
		"""
		Takes all buy and sell orders for all symbols as an input,
		and outputs a list of orders to be sent
		"""
        result = {}
        return result
```

**TradingState class**
```python
Time = int
Symbol = str
Product = str
Position = int
Observation = int

class TradingState(object):
    def __init__(self,
                 timestamp: Time,
                 listings: Dict[Symbol, Listing],
                 order_depths: Dict[Symbol, OrderDepth],
                 own_trades: Dict[Symbol, List[Trade]],
                 market_trades: Dict[Symbol, List[Trade]],
                 position: Dict[Product, Position],
                 observations: Dict[Product, Observation]):
        self.timestamp = timestamp
        self.listings = listings
        self.order_depths = order_depths
        self.own_trades = own_trades
        self.market_trades = market_trades
        self.position = position
        self.observations = observations
        
    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True)
```

**Trade class**
```python
Symbol = str
UserId = str

class Trade:
    def __init__(self, symbol: Symbol, price: int, quantity: int, buyer: UserId = None, seller: UserId = None, timestamp: int = 0) -> None:
        self.symbol = symbol
        self.price: int = price
        self.quantity: int = quantity
        self.buyer = buyer
        self.seller = seller
        self.timestamp = timestamp

    def __str__(self) -> str:
        return "(" + self.symbol + ", " + self.buyer + " << " + self.seller + ", " + str(self.price) + ", " + str(self.quantity) + ", " + str(self.timestamp) + ")"

    def __repr__(self) -> str:
        return "(" + self.symbol + ", " + self.buyer + " << " + self.seller + ", " + str(self.price) + ", " + str(self.quantity) + ", " + str(self.timestamp) + ")" + self.symbol + ", " + self.buyer + " << " + self.seller + ", " + str(self.price) + ", " + str(self.quantity) + ")"
```
**OrderDepth class**
```python
class OrderDepth:
    def __init__(self):
        self.buy_orders: Dict[int, int] = {}
        self.sell_orders: Dict[int, int] = {}
```

**Order class**
```python
Symbol = str

class Order:
    def __init__(self, symbol: Symbol, price: int, quantity: int) -> None:
        self.symbol = symbol
        self.price = price
        self.quantity = quantity

    def __str__(self) -> str:
        return "(" + self.symbol + ", " + str(self.price) + ", " + str(self.quantity) + ")"

    def __repr__(self) -> str:
        return "(" + self.symbol + ", " + str(self.price) + ", " + str(self.quantity) + ")"
```
