# rock-paper-scissors

![visitors](https://visitor-badge.glitch.me/badge?page_id=SouravJohar.visitor-badge)

https://youtu.be/0uSA3xyXlwM 

An AI to play the Rock Paper Scissors game

## Requirements
- Python 3
- Keras
- Tensorflow
- OpenCV

## Set up instructions
1. Clone the repo.
```sh
$ git clone https://github.com/SouravJohar/rock-paper-scissors.git
$ cd rock-paper-scissors
```

2. Install the dependencies
```sh
$ pip install -r requirements.txt
```

3. Gather Images for each gesture (rock, paper and scissors and None):
In this example, we gather 200 images for the "rock" gesture
```sh
$ python3 gather_images.py rock 200
```

4. Train the model
```sh
$ python3 train.py
```

5. Test the model on some images
```sh
$ python3 test.py <path_to_test_image>
```

6. Play the game with your computer!
```sh
$ python3 play.py
```
