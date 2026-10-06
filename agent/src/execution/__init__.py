"""Paper-trading execution bridge: verdict -> guarded plan -> bracket order.

Paper-only by construction. Live trading must go through ``src.live.sdk_order_gate``
and an explicit user-committed mandate; nothing here can reach a live account.
"""
