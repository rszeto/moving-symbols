\mainpage Moving Symbols API

# Entry Point

The `moving_symbols` package provides the `MovingSymbolsEnvironment` class, which is the one you
should be using to generate Moving Symbols videos. Click the link below to jump to the class' API:

moving_symbols.moving_symbols.MovingSymbolsEnvironment

# Tiny Example

The following code snippet puts the frames of one Moving Symbols video into a list:

```python
    from moving_symbols import MovingSymbolsEnvironment

    env = MovingSymbolsEnvironment(params, seed)

    all_frames = []
    for _ in xrange(num_frames):
        frame = env.next()
        all_frames.append(np.array(frame))
```

# MovingSymbolsEnvironment as a Publisher

MovingSymbolsEnvironment publishes messages corresponding to the initialization and state of each
symbol at all time steps. The following code snippet shows an example where a subscriber collects
all the published messages:

```python
    from moving_symbols import MovingSymbolsEnvironment

    class Subscriber:
        def process_message(self, message):
            print(message)

    sub = Subscriber()
    env = MovingSymbolsEnvironment(params, seed, initial_subscribers=[sub])

    all_frames = []
    for _ in xrange(num_frames):
        frame = env.next()
        all_frames.append(np.array(frame))
```

A subscriber can be added after the MovingSymbolsEnvironment instance is created by calling
moving_symbols.moving_symbols.MovingSymbolsEnvironment.add_subscriber. However, that subscriber
will not have access to the initialization messages, so setting `initial_subscribers` in the
constructor is recommended.