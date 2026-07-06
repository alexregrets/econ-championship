"""Business operations that coordinate the pure core with the database.

Services are the seam between :mod:`core` (deterministic logic) and :mod:`db`
(persistence): they load state through repositories, run core computations, and
write results back. Handlers, the API, and the dev-shell call services, not
repositories or the engine directly.
"""
