#!/usr/bin/env python3

from eaglemk4_nn_controller.controller import Controller


if __name__ == "__main__":
    c = Controller()
    try:
        c.run()
    except KeyboardInterrupt:
        c.close()
