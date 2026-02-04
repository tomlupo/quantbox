# quantbox (core)

Core runtime + plugin registry for **QuantBox**.

This package contains:
- plugin contracts (`quantbox.contracts`)
- entry-point discovery registry (`quantbox.registry`)
- artifact store (`quantbox.store`)
- runner (`quantbox.runner`)
- CLI (`quantbox.cli`)

Core intentionally contains **no strategy logic** and **no broker-specific code**.
