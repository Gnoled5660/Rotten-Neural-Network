My Goal here is to build a neural network from scratch and train it on youth slangs aka. brainrot. I have never built a neural network, so my goal is to understand how it works. 

After doing some research, I have categorized everthing into phases that needs to be done. Some phases or the code of these phases are not necessary in the end, but they are important for the general understanding. 
1. Autograd for scalars: A Value class that records every calculation in a computation graph and computes derivatives automatically using the chain rule. It can be used to train small neural network. 
2. Tensor autograd with NumPy: Improve the autograd from single numbers to whole matrices, including matrix multiplication, broadcasting and a numerically stable softmax. 
3. Network building blocks + MNIST: Code reusable layers, a loss function and optimizers. 
4. Language model basics: Train the first models that can predict the next character in a text.
5. BPE Tokenizer: A tokenizer that splits text into words and word pieces.
6. Transformer: Self-attention, multiple heads, layer normalization and residual connections.
7. Full training: Train the model with 5 million parameters such as Tiny Stories. 
8. Chat fine-tuning: Train the model on dialogues in a fixed format -> learns to reply to messages instead of continuning text.
9. Youth Slang Persona:
-> last but not least, a web interface to chat with the ai.