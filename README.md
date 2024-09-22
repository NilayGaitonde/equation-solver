# Mathematical Equation Recognition and Solving
In S4 E12 of The Big Bang Theory, Leonard comes up with an idea for a smartphone app to identify differential equations. Now I saw this episode recently and have since started working on this app. 
Currently it just recognizes basic addition, multiplication, subtraction and division equations but I plan to expand it to also include more complex equtions. I also currently have my internship going on so a bit tied up with that as well. 
## Key Features
- **High Accuracy**: Achieves an impressive 97% training accuracy in recognizing handwritten mathematical symbols and equations.
- **Image Processing**: Utilizes OpenCV for efficient image preprocessing and contour detection.
- **Deep Learning Model**: Implements a custom resnet based neural network architecture for digit and symbol recognition.
- **Equation Parsing**: Converts recognized symbols into solvable mathematical expressions.
- **Solves equation**: The application recognises the equation and solves it.
- **CUDA Support**: Trained using CUDA for accelerated performance, with CPU inference capabilities for broader accessibility.

## Important Links:
- **Weights and Biases**: Check out the weights and biases report to see how the model training progress went

## Future Improvements
- Implement more complex mathematical operations recognition
- Enhance the user interface for better interactivity
- Optimize the model for mobile deployment

## Docker Support
I've also included a dockerfile for easy deployment. To build and run the docker container:
```docker build -t maths-app .```
To run it:
```docker run -p 8501:8501 maths-app```
