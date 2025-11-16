
import logging

from .game2048 import Game2048, Action

def _action_from_input(s: str) -> Action | None:
    s = s.strip().lower()
    if not s:
        return None

    key_map = {
        "w": 0,
        "d": 1,
        "s": 2,
        "a": 3,
        "k": 0,
        "l": 1,
        "j": 2,
        "h": 3,
    }

    if s in key_map:
        return key_map[s]

    if s.isdigit():
        n = int(s)
        if n in (0, 1, 2, 3):
            return n

    return None


def main():
    game = Game2048(size=4)
    game.reset(1)

    while True:
        print(game.render())
        if game._is_done():
            print("GAME OVER")
            break

        user_input = input("Operation").strip().lower()

        if user_input and user_input in ("q"):
            print("QUIT")
            break

        action = _action_from_input(user_input)
        if action is None:
            print("error input")
            continue

        changed, state, merged, done = game.step(action)

        if not changed:
            print("unchanged")

        if merged:
            print(game.render())

        if done:
            print(game.render())
            print("GAME OVER")
            break
        
def log_setup():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler("game2048.log", encoding="utf-8"),
        ],
    )

    logger = logging.getLogger(__name__)


if __name__ == "__main__":
    log_setup()
    main()
