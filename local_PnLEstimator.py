from __future__ import annotations

from datamodel import Order, Symbol


class ProfitsAndLossesEstimator:
    def __init__(self, products: list[str]) -> None:
        self.profits_and_losses: dict[Symbol, int] = {
            product: 0 for product in products
        }

    def update(self, order: Order, partial: int | None = None) -> None:
        """
        Updates the P&L for the given order.
        Partial is an argument that is used when the order is partially filled.
        """
        quantity = partial if partial is not None else order.quantity
        self.profits_and_losses[order.symbol] = self.profits_and_losses[
            order.symbol
        ] + order.price * (  # current P&L for this symbol
            -quantity
        )  # if we buy (positive order.quantity) we lose money, if we sell (negative order.quantity) we gain money

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
