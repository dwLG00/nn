# neural net

Small neural net framework, written using numpy. Intended as a learning exercise.
I did all the vector math by hand. It took way too long. I need to brush up on my vector calculus.
Right now the framework supports fully connected or Sigmoid activation layers, and a simple dot product output layer.
The framework isn't very fast. It took about 22 seconds to run 10,000 training loops for a 2 -> 5 -> 4 network, with each training batch being 100 datapoints large. I don't plan to optimize the speed very much because it's just a learning exercise for me.
Don't use this for anything serious!
