
## Natural Language Processing

- The techniques for computer programs to classify, understand and generate human language.
- Some Applications:
	- Machine translation(such as Google Translate)
	- NLG (Natural Language Generation)
	- Information Retrieval
	- Spam Filters
	- Sentiment Analysis
	- Chatbots
	- Linguistic analysation
	- Social Science Analysation

### Machine Learning

```An agent is learning if it improves its performance after making observations about the world. Learning can range from the trivial, such as jotting down a shopping list, to the profound, as when Albert Einstein inferred a new theory of the universe. When the agent is a computer, we call it machine learning: a computer observes some data, builds a model based on the data, and uses the model as both a hypothesis about the world and a piece of software that can solve problems.```

![[Pasted image 20230805185141.png]]

So, Machine Learning a subset of AI where algorithms learn from data to make predictions or decisions without explicit programming for the task.
### Classification vs Regression

```When the output is one of a finite set of values (such as sunny/cloudy/rainy or true/false), the learning problem is called classification. When it is a number (such as tomorrow’s temperature, measured either as an integer or a real number), the learning problem has the (admittedly obscure) name regression.```
### Deep Learning

```Deep learning is a broad family of techniques for machine learning in which hypotheses take the form of complex algebraic circuits with tunable connection strengths. The word “deep” refers to the fact that the circuits are typically organized into many layers, which means that computation paths from inputs to outputs have many steps. Deep learning is currently the most widely used approach for applications such as visual object recognition, machine translation, speech recognition, speech synthesis, and image synthesis; it also plays a significant role in reinforcement learning applications```

![[Pasted image 20230805184555.png]]

### Neural Networks

```Deep learning has its origins in early work that tried to model networks of neurons in the brain (McCulloch and Pitts, 1943) with computational circuits. For this reason, the networks trained by deep learning methods are often called neural networks, even though the resemblance to real neural cells and structures is superficial.```

![[Pasted image 20230805190529.png]]

So, Neural networks are a foundational concept in deep learning and are inspired by the workings of the human brain. With reference the above slide, using the formula `y1 = W1*x + b1` as a starting point, we can try to understand how they work.

1. **Basic Structure**:
   Neural networks consist of layers of interconnected nodes or neurons. They are typically divided into three types of layers:
   - **Input Layer**: Represents the input features of a dataset.
   - **Hidden Layer(s)**: Intermediate layers between input and output. They transform the inputs into something that the output layer can use.
   - **Output Layer**: Produces the prediction for the given input.

2. **Weights and Bias**:
   Using our formula, `W1` represents the weight and `b1` represents the bias. Every connection between two neurons has a weight, which determines the strength or importance of that connection. The bias, on the other hand, is like an intercept in linear regression, ensuring that even when all input features are zero, there's a non-zero output.

   - `y1 = W1*x + b1` 
   
   Here, `W1*x` is a weighted sum of the inputs, and `b1` adjusts the output.

3. **Activation Function**:
   After computing the weighted sum and adding bias, the result is passed through an activation function. This function introduces non-linearity to the model, enabling it to learn complex patterns. Common activation functions include the sigmoid, tanh, and ReLU.

4. **Prediction**:
   Once the data is passed through the neural network, we get an output, which is the network's prediction. In a classification task, the output could represent the probability of belonging to a particular class.

5. **Label**:
   The label is the actual truth or target value in supervised learning. For instance, if you're classifying images of cats and dogs, the label tells you whether an image is of a cat or a dog.

6. **Loss**:
   The difference between the prediction and the label is quantified using a loss function. Common loss functions include Mean Squared Error for regression tasks and Cross-Entropy for classification tasks. The goal during training is to minimize this loss.

7. **Gradient and Backpropagation**:
   To adjust weights and biases to minimize the loss, we compute the gradient of the loss function with respect to each weight and bias. The gradient points in the direction of the steepest increase in the loss. 

   Backpropagation is an algorithm that computes these gradients in an efficient manner. It works by propagating the gradient of the loss backward through the network.

8. **Updating Weights using Gradients**:
   Once gradients are computed, they are used to update the weights and biases, typically using optimization algorithms like Stochastic Gradient Descent (SGD). The formula might look something like:
   
   `W1 = W1 - learning_rate * gradient`

   Here, the learning rate is a small positive number that determines the step size in the direction opposite to the gradient.
   
In essence, a neural network works by repeatedly adjusting its weights and biases based on the error of its predictions, using the backpropagation algorithm and an optimization method like SGD, until the error reaches an acceptably low level or stops improving.

![[Pasted image 20230805191434.png]]

### Loss Function

![[Screenshot 2023-08-05 at 7.22.35 pm.png]]

The concept revolves around gradient descent where the gradient (or derivative) of the loss function with respect to the network's weights is computed. This gradient essentially tells how much the weights should be adjusted to minimize the loss.
### Gradient and Backpropagation:

![[Pasted image 20230805192858.png]]

1. **Understanding Gradients**:
In the context of neural networks, a gradient is a vector that represents the direction and rate of fastest increase of a function. In this case, the function is the loss function, and it tells us how much the loss will change if we adjust our parameters (weights and biases) by a tiny amount. Gradients are computed using derivatives, which give us a measure of the rate of change.

2. **Partial Derivatives**:
Since neural networks have multiple parameters, we need to understand how the loss changes concerning each parameter. This is where partial derivatives come in. A partial derivative represents the rate of change of a function concerning one of its variables, keeping the others constant.

For instance, if we have a function \( f(x, y) \), then the partial derivative with respect to \( x \) tells us how \( f \) changes as \( x \) changes, keeping \( y \) constant.

3. **Chain Rule & Neural Networks**:
In neural networks, the loss function's output is a composite function of several operations involving weights, biases, and activations. The chain rule is a fundamental principle in calculus that allows us to compute the derivative of such composite functions.

Mathematically, if we have a composition of functions, say \( y = g(f(x)) \), then the derivative of \( y \) with respect to \( x \) is given by:

![[Pasted image 20230805193246.png]]

This rule becomes crucial for backpropagation, as we break down the gradient computation layer by layer.

**Backpropagation**:

1. **From Loss Function to Input Direction**:
Backpropagation is essentially applying the chain rule from calculus in a systematic manner, layer by layer, from the output (loss function) back to the input. This is why it's called "back" propagation.

 2. **Storing Values for Each Step for Quick Read (Caching)**:
During the forward pass, when input data is passed through the network, and outputs are generated layer by layer, we store (or cache) the intermediate values (like input, output, and activations of each layer). This caching is crucial because, during backpropagation, we use these cached values to compute the gradients without having to recompute them. This considerably speeds up the learning process.

3. **Using the Parameters from Forward Propagation**:
During backpropagation, the gradients for each layer are computed concerning the loss function. The gradients signify how much the parameters (weights and biases) of that specific layer need to be adjusted to reduce the overall error. Using cached values from the forward pass and the chain rule, these gradients are calculated.

Let's break down the process for a simple feedforward neural network with one hidden layer:

1. **Forward Pass**:
    - Input \( x \) is passed through the hidden layer, generating an output using weights \( W_1 \), biases \( b_1 \), and an activation function \( f \): \( h = f(W_1 x + b_1) \).
    - This output \( h \) is then passed to the output layer with weights \( W_2 \) and biases \( b_2 \) to get the final prediction: \( y' = W_2 h + b_2 \).
    - Compute the loss, say \( L \), between the predicted \( y' \) and the actual output \( y \).

2. **Backward Pass**:
    - Compute the gradient of the loss with respect to \( y' \).
    - Use the chain rule to compute the gradients of the loss concerning \( W_2 \) and \( b_2 \).
    - Again, using the chain rule, compute the gradient of the loss concerning the output of the hidden layer \( h \).
    - Compute the gradients of the loss concerning \( W_1 \) and \( b_1 \) using the chain rule.

After computing the gradients, we update the weights and biases using these gradients, typically employing optimization techniques like Gradient Descent. This entire process is repeated for several iterations (or epochs) until the network converges to an optimal set of parameters.

### Training Process

![[Pasted image 20230805193509.png]]

![[Pasted image 20230805193525.png]]

### References

1. Artificial Intelligence: A Modern Approach, 4th US ed.
