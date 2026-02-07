__all__ = [
    'get_binance_client',
    'generate_portfolio_orders',
    'execute_orders',
    'initialize_paper_trading',
    'send_telegram_notifications'
]

from .binance import get_binance_client
from .orders import generate_portfolio_orders, execute_orders
from .paper_trading import initialize_paper_trading

# Telegram is optional (needs matplotlib, tabulate)
try:
    from .telegram import send_telegram_notifications
except ImportError:
    def send_telegram_notifications(*args, **kwargs):
        pass  # No-op if deps missing