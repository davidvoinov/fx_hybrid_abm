class Order:
    """
    Order contains all relevant information about order, it can be of two types: bid, ask. Supports binary comparison
    operations of price among all pairs of order types according to the logic:

    **better-offer < worse-offer**
    """
    order_id = 0

    def __init__(self, price, qty, order_type, trader_link=None, ttl=None):
        # Properties
        self.price = price
        self.qty = qty
        self.order_type = order_type
        self.trader = trader_link
        # Anonymous exchange support remains anonymous for book ownership,
        # but may settle against an explicit external-sector account.  This
        # avoids creating/destroying cash and base whenever a background
        # order is hit.
        self.settlement_account = None
        self.order_id = Order.order_id
        self.ttl = ttl  # ticks until expiry (None = GTC)
        self.reserved_cash = 0.0
        self.reserved_assets = 0.0
        # Fills generated while this order is aggressive.  The execution
        # record needs the passive owner in order to distinguish bank-dealer
        # liquidity from non-bank and anonymous liquidity.
        self.fills = []

        # Connections
        self.left = None
        self.right = None
        Order.order_id += 1

    def __lt__(self, other) -> bool:
        if self.order_type != other.order_type:
            if self.order_type == 'bid':
                return self.price > other.price
            if self.order_type == 'ask':
                return self.price < other.price

        if self.order_type == 'bid':
            return self.price > other.price
        if self.order_type == 'ask':
            return self.price < other.price

    def __le__(self, other) -> bool:
        if self.order_type != other.order_type:
            if self.order_type == 'bid':
                return self.price >= other.price
            if self.order_type == 'ask':
                return self.price <= other.price

        if self.order_type == 'bid':
            return self.price >= other.price
        if self.order_type == 'ask':
            return self.price <= other.price

    def __gt__(self, other) -> bool:
        if self.order_type != other.order_type:
            if self.order_type == 'bid':
                return self.price < other.price
            if self.order_type == 'ask':
                return self.price > other.price

        if self.order_type == 'bid':
            return self.price < other.price
        if self.order_type == 'ask':
            return self.price > other.price

    def __ge__(self, other) -> bool:
        if self.order_type != other.order_type:
            if self.order_type == 'bid':
                return self.price <= other.price
            if self.order_type == 'ask':
                return self.price >= other.price

        if self.order_type == 'bid':
            return self.price <= other.price
        if self.order_type == 'ask':
            return self.price >= other.price

    def __repr__(self) -> str:
        return f'{self.order_type} (price={self.price}, qty={self.qty})'

    def to_dict(self) -> dict:
        return {'price': self.price, 'qty': self.qty, 'order_type': self.order_type,
                'trader_link': self.trader}

    @classmethod
    def from_dict(cls, order_dict):
        return Order(order_dict['price'], order_dict['qty'], order_dict['order_type'], order_dict.get('trader_link'))


class OrderIter:
    """
    Iterator class for OrderList
    """
    def __init__(self, order_list):
        self.order = order_list.first

    def __iter__(self) -> 'OrderIter':
        # Iterator protocol: an iterator must return itself, otherwise
        # generator expressions / comprehensions over an OrderList fail
        # (they call iter() a second time on the OrderIter), even though a
        # plain for-loop works.
        return self

    def __next__(self) -> Order:
        if self.order:
            next_order = self.order
            self.order = self.order.right
            return next_order
        raise StopIteration


# noinspection DuplicatedCode
class OrderList:
    """
    OrderList is implemented as a doubly linked list. It preserves the same order type inside,
    all orders are sorted according to best-offer -> worst-offer dynamically.

    remove, append, push: complexity O(1)

    insert, fulfill (for large qty): complexity O(n)
    """
    def __init__(self, order_type: str):
        self.first = None
        self.last = None
        self.order_type = order_type

    def __iter__(self) -> OrderIter:
        return OrderIter(self)

    def __bool__(self):
        return self.first is not None and self.last is not None

    def __len__(self):
        n = 0
        for order in self:
            n += 1
        return n

    def to_list(self) -> list:
        return [order.to_dict() for order in self]

    def remove(self, order: Order):
        if order.order_type != self.order_type:
            raise ValueError(f'Wrong order type! OrderList: {self.order_type}, Order: {order.order_type}')
        if order == self.first:
            self.first = order.right
        if order == self.last:
            self.last = order.left

        if order.left:
            order.left.right = order.right
        if order.right:
            order.right.left = order.left

        order.left = None
        order.right = None

    def append(self, order: Order):
        # If wrong order type to insert
        if order.order_type != self.order_type:
            raise ValueError(f'Wrong order type! OrderList: {self.order_type}, Order: {order.order_type}')

        if not self.first:
            self.first = order
            self.last = order
            return

        self.last.right = order
        order.left = self.last
        self.last = order

    def push(self, order: Order):
        """
        Insert order in the beginning

        :param order: Order
        :return: void
        """
        # If wrong order type to insert
        if order.order_type != self.order_type:
            raise ValueError(f'Wrong order type! OrderList: {self.order_type}, Order: {order.order_type}')

        if not self.first:
            self.first = order
            self.last = order
            return

        self.first.left = order
        order.right = self.first
        self.first = order

        if order.order_type != self.order_type:
            raise ValueError(f'Wrong order type! OrderList: {self.order_type}, Order: {order.order_type}')

    def insert(self, order: Order):
        """
        Inserts order preserving best-offer -> worst-offer

        :param order: Order
        :return: void
        """
        # If wrong order type to insert
        if order.order_type != self.order_type:
            raise ValueError(f'Wrong order type! OrderList: {self.order_type}, Order: {order.order_type}')

        # If empty
        if self.first is None:
            self.append(order)
            return

        # Insert order in the beginning.
        #
        # The comparison is strict, so an order arriving at a price that is
        # already in the book goes behind what is resting there rather than
        # in front of it. With the non strict form the queue ran last in
        # first out at a given price: the newest quote was filled first and
        # posting early carried no advantage. That is neither price time
        # priority nor any allocation rule the venue publishes, and it feeds
        # straight into fill probability, adverse selection, the life of an
        # order, dealer profit and loss, and how fast inventory is built and
        # worked off.
        if order < self.first:
            order.right = self.first
            self.first.left = order
            self.first = order
            return

        # Insert order in the middle, again strictly, so the new order lands
        # after every order already standing at its own price.
        for val in self:
            if order < val:
                # If only 1 order in self
                if self.first == self.last:
                    self.push(order)

                order.left = val.left
                order.right = val
                order.left.right = order
                order.right.left = order
                return

        # Insert to the end
        self.append(order)

    def fulfill(self, order: Order, t_cost: float) -> Order:
        if order.order_type == self.order_type:
            raise ValueError(f'Wrong order type! OrderList: {self.order_type}, Order: {order.order_type}')

        for val in self:
            if order.qty == 0:
                break
            if val > order:
                break

            # Solve orders
            order_qty_before = order.qty
            val_qty_before = val.qty
            tmp_qty = min(order.qty, val.qty)  # Quantity traded currently
            tmp_price = val.price  # Price traded
            val.qty -= tmp_qty
            order.qty -= tmp_qty

            maker = val.trader
            order.fills.append({
                'quantity': float(tmp_qty),
                'price': float(tmp_price),
                'maker_id': getattr(maker, 'id', None),
                'maker_type': getattr(maker, 'type', 'background') if maker is not None else 'background',
            })

            # Solve cash and assets
            if order.order_type == 'bid':
                if order.trader is not None or order.settlement_account is not None:
                    self._settle_trader_fill(order, tmp_qty, tmp_price, t_cost,
                                             is_buy=True, qty_before=order_qty_before)
                if val.trader is not None or val.settlement_account is not None:
                    self._settle_trader_fill(val, tmp_qty, tmp_price, t_cost,
                                             is_buy=False, qty_before=val_qty_before)

            if order.order_type == 'ask':
                if order.trader is not None or order.settlement_account is not None:
                    self._settle_trader_fill(order, tmp_qty, tmp_price, t_cost,
                                             is_buy=False, qty_before=order_qty_before)
                if val.trader is not None or val.settlement_account is not None:
                    self._settle_trader_fill(val, tmp_qty, tmp_price, t_cost,
                                             is_buy=True, qty_before=val_qty_before)

            # Clear remaining
            if val.qty == 0:
                self.remove(val)

        return order

    @staticmethod
    def _settle_trader_fill(order: Order, fill_qty: float, fill_price: float,
                            t_cost: float, is_buy: bool, qty_before: float):
        trader = order.trader or getattr(order, 'settlement_account', None)
        if trader is None:
            return

        if hasattr(trader, 'apply_fill'):
            trader.apply_fill(order, fill_qty, fill_price, t_cost, is_buy, qty_before)
            return

        gross = fill_price * fill_qty
        if is_buy:
            trader.cash -= gross * (1 + t_cost)
            trader.assets += fill_qty
        else:
            trader.cash += gross * (1 - t_cost)
            trader.assets -= fill_qty

    @classmethod
    def from_list(cls, order_list, sort=False):
        order_list = [Order(order['price'], order['qty'], order['order_type'],
                            order.get('trader_link')) for order in order_list]
        order_list_obj = OrderList(order_list[0].order_type)
        if sort:
            for order in order_list:
                order_list_obj.insert(order)
        else:
            for order in order_list:
                order_list_obj.append(order)
        return order_list_obj
