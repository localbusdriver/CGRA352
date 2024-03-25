import cv2
from PIL import Image
from core import Core
from completion import Completion
from challenge import Challenge

def run() -> None:
    flower = cv2.imread("./assets/Flower.jpg")
    flower_grayscale = cv2.imread("./assets/Flower.jpg", cv2.IMREAD_GRAYSCALE)
    building = cv2.imread("./assets/Building.jpg", cv2.IMREAD_GRAYSCALE)

    core = Core(flower)
    core.display()

    comp = Completion(flower_grayscale)
    comp.display()

    chal = Challenge(building)
    chal.display()

    return

if __name__ == "__main__":
    run()