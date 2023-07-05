# What is AI?
Artificial intelligence is an umbrella term that in general describes the broad research and applications that try to enable intelligent behaviour by machines. Any device that can perceive inputs from its environment and take actions accordingly is applying AI. Using AI machines can mimic human cognitive intelligence. To mimic human cognitive intelligence, AI relies on Knowledge engineering - where we need to derive and provide contextual information from real world to the machine to solve the contextual problem. For example, to solve complex language problems, AI would need to know the complex relationships between letters, words, sentences, grammar and other dimensions of what we call as "human language". The knowledge engineering aspect of AI would need to derive and provide massive amounts of samples that show this relationship in a good variety so that the resulting AI can mimic general variety of Natural Language Intelligence.

Massive DATA + Complex CODE ≠ AI
Complex contextual knowledge representations + Algorithms + Experimentation = Modelling AI 
i.e., Mimicking decisions and outcomes in the same way human would in the same context (represented in the input data as features/dimensions/properties that are significant in each problem space)

So, aim of Modelling in AI is to transfer the human like expertise to a software program by feeding it real-world data that would make the software "Intelligent" enough to perceive information from the given data to produce same outcomes as a human would; given the same data in the context of the problem.

Modelling hence will not be a one-time activity rather an ongoing cycle of refinement by providing more data or tweaking the software algorithm until its decisions are as close to a human as possible for the given context.

# What is generative AI?

# What are the other sub-areas in AI?
Machine Learning and Deep Learning are subsets of AI that focusses on algorithms that enable modelling approaches which can produce state of the art performances comparable to humans.
   |--- Supervised Learning
         |       |--- Regression
         |       |--- Classification
         |
         |--- Unsupervised Learning
         |       |--- Clustering
         |       |--- Dimensionality Reduction
         |       |--- Anomaly Detection
         |
Contrastive Learning: Contrastive learning is a type of unsupervised learning where the model learns by contrasting positive samples (data from the same class or similar) against negative samples (data from different classes or dissimilar). This approach encourages the model to create meaningful representations that differentiate between similar and dissimilar data.
         |--- Semi-Supervised Learning
         |
         |--- Reinforcement Learning
         |--- Transfer Learning
The basic idea behind transfer learning is that the features learned by a model on one task can be useful for another task. Instead of starting the training process from scratch and initializing the model randomly, transfer learning allows us to initialize the model with the pre-trained weights, which already capture useful patterns and information from the original task.
During fine-tuning, the earlier layers of the pre-trained model are often frozen or have their learning rates reduced to preserve the previously learned features. The later layers or task-specific layers are typically modified or replaced to adapt the model to the new task. By doing so, the model can effectively learn task-specific patterns while still benefiting from the generic features learned in the pre-training stage.
         |--- Deep Learning
                 |--- Convolutional Neural Networks (CNNs)
CNNs have been revolutionary in computer vision tasks, including image classification, object detection, and image segmentation.
CNNs are best suited for processing grid-like structured data, such as images, where local patterns and spatial relationships are crucial. They leverage convolutional and pooling layers to extract hierarchical features, allowing them to capture visual patterns and generalize well across different images. CNNs are feed-forward models and do not have explicit memory to retain information from previous inputs.
                 |--- Recurrent Neural Networks (RNNs) 
RNNs are commonly used in tasks such as natural language processing (NLP), speech recognition, sentiment analysis, and machine translation.
RNNs are designed to handle sequential data, where the order and temporal dependencies matter. RNNs use recurrent connections to maintain memory across time steps, enabling them to process and predict sequences. They are effective in tasks involving text, time series data, or any sequential information. However, RNNs may face challenges in capturing long-term dependencies due to the vanishing or exploding gradient problem.
                 |--- Generative Adversarial Networks (GANs)
GANs are primarily used for tasks like image generation, video synthesis, and data generation.
The goal of GANs is to train the generator to generate data that is indistinguishable from real data, while the discriminator aims to become better at distinguishing between real and synthetic data. The two networks are trained together in a competitive fashion, with the generator trying to fool the discriminator, and the discriminator trying to accurately classify the data.
                 |--- Autoencoders
Autoencoders can be used for tasks like image denoising, anomaly detection, and feature extraction. They are also used as a pretraining step in deep learning architectures, where the encoder's learned weights are used to initialize the weights of another neural network. Variable Autoencoders
                 |--- Transformers
Transformers process data in parallel.
Transformers rely on an attention mechanism, which allows the model to selectively focus on different parts of the input data. Instead of using recurrent connections, Transformers use self-attention to capture the relationships between different positions within the input sequence. This self-attention mechanism allows the model to weigh the importance of different words or elements in the sequence when making predictions.
Transformers have shown impressive performance in tasks such as machine translation, text generation, and document classification. They have the advantage of parallel processing, which allows them to handle longer sequences more efficiently than RNNs. Transformers have become particularly popular with the introduction of the "Transformer" architecture, as seen in models like BERT, GPT, and T5.


# Machine Learning
Machine Learning = Cycle of evaluation and optimisation (Data, Processing the data, Learned patterns in data, Predictions or decisions based on patterns) 
https://ckaestne.medium.com/machine-learning-in-production-book-overview-63be62393581

Machine learning is a broad field that focuses on developing algorithms and models that can learn patterns and make predictions or decisions based on data. It involves training models on processed data, extracting meaningful features, and using various algorithms to make predictions or make decisions. Machine learning algorithms can be categorized into supervised learning, unsupervised learning, and reinforcement learning.
Often used where traditional software programming cannot model a solution for a given complex problem because the software will get complex quickly. Instead, machine learning is used to model AI into software (the ML Model) that can then learn patterns of relationships and outcomes from input data to get to predictions that helps solve the complex problem.
Predict
Optimise
Group
Classify
**Parameters:**
Parameters in ML are variables that determine the behavior and characteristics of a model. They are learned during the training process by adjusting their values to optimize the model's performance. Parameters capture the underlying relationships and patterns in the training data, allowing the model to make predictions or decisions.
In ML, parameters can refer to different entities depending on the type of model:
Linear Models: In linear models such as linear regression or logistic regression, the parameters are the coefficients assigned to each input feature. These coefficients control the magnitude and direction of the impact that each feature has on the model's predictions.
Decision Trees: In decision trees, parameters correspond to the splitting rules at each internal node. These rules determine the conditions based on which the input data is partitioned into different branches of the tree.
Support Vector Machines: In support vector machines (SVMs), parameters include the weights assigned to each data point (support vectors) and the bias term. These parameters define the position and orientation of the decision boundary.
**Nodes:**
In ML, nodes, also known as units or neurons, are computational units that receive inputs, perform calculations, and produce an output. Nodes can be present in various types of ML models, such as neural networks, decision trees, or support vector machines.
In neural networks, nodes are the fundamental building blocks. They are organized into layers, and each node performs a specific computation using its inputs and internal parameters. Nodes apply a mathematical operation (e.g., a weighted sum or a non-linear activation) to the input data and pass the result to subsequent nodes or the final output.
In other models like decision trees or support vector machines, nodes represent decision points or support vectors that contribute to the model's decision-making process. Each node is associated with a specific rule or condition based on which the model determines the appropriate outcome or prediction.
Overall, parameters and nodes are crucial elements in ML models. Parameters are learned during training and capture the relationships within the data, while nodes process inputs and contribute to the model's computations and decision-making process.

**1. Problem Definition**
Problem = (Input Data, Desired Output, Objective)
This equation represents the initial step in AI modelling, where we define the problem to be solved. The problem is characterized by the input data required for the model, the desired output it should produce, and the overarching objective or goal of the AI system.

2. Data Collection and Preprocessing
Processed Data = Preprocess (Input Data)
Processed Data -> separate into Training Data and Evaluation Data
Here, we gather relevant data for the model and then preprocess it to ensure it is in a suitable format for training. Preprocessing involves tasks such as data cleaning, normalization, feature extraction, and handling missing values. Feature extraction is specifying what features from the given data should be trained on to learn the patterns. Further the processed data is divided into training data and evaluation data so that the model can be trained on a subset of available data then evaluated on a subset that the model has not seen so far.

3. Model Selection and Training
Best Model = Evaluate (Performance Metrics, Algorithms, Training Data)
Evaluate different algorithms using appropriate performance metrics to select the best model for the given problem.
Trained Model with learned parameters = Train (Best Model, Training Data)
We then train the chosen model using the pre-processed data, enabling it to learn patterns and relationships within the data.

4. Model Evaluation and Fine-tuning
Optimised Model = FineTune (Trained Model, Evaluation Data, Evaluation Metrics)
After training the model, we evaluate its performance on a separate dataset to assess its effectiveness. If the model's performance is unsatisfactory, we use the feedback from evaluation to conduct fine-tuning, adjusting hyperparameters or making modifications to improve its performance.

5. Model Deployment and Monitoring
Deployed Model = Deploy (Optimized Model)
Monitoring = Monitor (Deployed Model)
In this final step, the optimized model is deployed to the real-world environment, where it interacts with users or applications to provide predictions or solutions. Concurrently, the model is continuously monitored to ensure its ongoing effectiveness and to detect and address any issues that may arise in production. Monitoring involves evaluating the model's performance over time and, if needed, retraining or updating the model to maintain its accuracy and relevance.

Deep Learning
(example of coloured rectangles vs shape that is rectangle that is not in same colour)
Is deep learning unsupervised approach?
Deep learning is a subset of machine learning. While both deep learning and machine learning are branches of artificial intelligence, they differ in terms of their approach and the types of problems they can solve.
Deep learning is a specialized approach within machine learning that leverages deep neural networks to learn hierarchical representations of data, while machine learning encompasses a broader set of techniques and algorithms for training models to make predictions or decisions based on data.
Deep learning utilizes artificial neural networks with multiple layers (hence the term "deep") to learn and extract hierarchical representations of the given data. Deep learning algorithms automatically learn these hierarchical representations on its own, without the need for specifying features manually as in other machine learning approaches(rather than being told what features to look at, deep learning nables the machine to define the features itself from the data). This is achieved by progressively auto-extracting higher-level features from raw input data. This allows deep learning models to discover complex patterns and relationships in the data. 
Deep neural networks can handle large amounts of data and learn more intricate and abstract representations, making them particularly effective for tasks such as image and speech recognition, natural language processing, and computer vision. However, deep learning models require a substantial amount of data and computational resources for training, as well as careful tuning of hyperparameters.

Node:
In a neural network, a node, also referred to as a neuron, is the fundamental building block. It represents a processing unit that receives input, performs computations, and produces an output. Each node typically applies a mathematical operation to the input data and passes the result to subsequent nodes.
Layers:
Layers in a neural network refer to groups of nodes that are arranged in a sequential manner. Each layer performs a specific function during the computation process. In a typical deep neural network, there are three main types of layers:
Neural Network:
A neural network is a computational model inspired by the structure and functionality of the human brain. It is composed of interconnected processing units called nodes or neurons. Neural networks are designed to simulate the learning and decision-making processes of the human brain by processing and transforming input data to produce desired outputs.

Input Layer:
The input layer is the first layer of a neural network. It receives the initial input data and passes it to the subsequent layers for processing. The number of nodes in the input layer is determined by the dimensions or features of the input data.
Hidden Layers:
Hidden layers are the intermediate layers between the input and output layers of a neural network. They are called "hidden" because their computations are not directly observable from the input or output. Hidden layers enable the network to learn complex representations and extract hierarchical features from the input data. Deep neural networks often consist of multiple hidden layers.
Output Layer:
The output layer is the final layer of a neural network. It produces the desired output or predictions based on the computations performed by the preceding layers. The number of nodes in the output layer depends on the nature of the problem the network aims to solve. For example, in a binary classification task, there might be a single node in the output layer representing the probability of belonging to one of the classes.

The connections between nodes in adjacent layers are represented by weights, which indicate the strength or importance of the connections. During the training process, these weights are adjusted to optimize the network's performance by minimizing the error between predicted and actual outputs.
Supervised Learning:
In supervised learning, a deep learning model is trained using labeled data, where the input samples are paired with their corresponding target or output values. The model learns to map the input data to the desired outputs by optimizing a specified loss or error function. Supervised learning is commonly used for tasks such as image classification, object detection, speech recognition, and natural language processing. Deep learning models like convolutional neural networks (CNNs) and recurrent neural networks (RNNs) are often employed in supervised learning scenarios.
Unsupervised Learning:
In unsupervised learning, the deep learning model learns patterns and structures from unlabeled data. It aims to discover inherent relationships and underlying representations within the data without explicit guidance from labeled examples. Unsupervised learning can be used for tasks such as clustering, dimensionality reduction, anomaly detection, and generative modeling. Deep learning models like autoencoders and generative adversarial networks (GANs) are popular approaches in unsupervised learning.

In ML, a significant amount of effort is often dedicated to feature engineering, where domain-specific features are manually designed and extracted from the data. These engineered features are then used as input to the ML model during training.
In DL, feature engineering is typically less explicit. Deep neural networks are capable of learning hierarchical representations of features directly from raw input data. Instead of handcrafting features, DL models learn to extract relevant features as part of the training process, allowing for more automated feature learning.

In ML, models typically have a fixed or limited number of parameters that need to be learned. These models often rely on handcrafted features engineered by domain experts. The training process involves finding optimal values for these parameters to minimize the error or loss.
In DL, models are composed of multiple layers of interconnected nodes (neurons) organized into deep neural networks. These networks can have a vast number of parameters, often in the millions or more. Deep learning models learn hierarchical representations and feature extraction directly from the raw data. The training process involves optimizing these numerous parameters to learn intricate and complex patterns.

Embeddings:
In ML, embeddings refer to the representation of data in a lower-dimensional space, where each data point is encoded as a vector of numbers. Embeddings capture the essential characteristics or features of the data in a more compact form. They are learned through training models on large datasets and are designed to capture meaningful relationships between data points.
Vectors:
In ML, vectors are mathematical objects that represent both magnitude and direction. They are one-dimensional arrays of numbers that can represent various attributes or features of a data point. Vectors can be used to represent different types of information, such as the numerical features of an image or the word frequencies in a text document. They are commonly used to perform mathematical operations and comparisons on data.
Tokens:
In ML, tokens refer to the individual units or elements into which a piece of data is divided. Tokens can represent various components depending on the context. In natural language processing, tokens are often words, but they can also be characters, subwords, or other linguistic units. Tokens are used to break down the input data into manageable pieces, allowing the ML model to process and analyze the data effectively.


Why now?
Turing Machine (1936): The concept of a universal machine, introduced by Alan Turing, laid the theoretical foundation for the development of modern computers and computing machines, which later became essential for AI research and applications.
Dartmouth Conference (1956): In the summer of 1956, the Dartmouth Conference marked the birth of AI as a field. Researchers including John McCarthy, Marvin Minsky, Allen Newell, and Herbert Simon convened to discuss the possibility of creating machines capable of exhibiting intelligence.
Symbolic AI and Expert Systems (1960s-1980s): In the 1960s, researchers focused on Symbolic AI, using formal rules and logic to mimic human reasoning. Expert systems, such as DENDRAL (1965) and MYCIN (1976), showcased the potential of AI by demonstrating expert-level knowledge in specific domains.
Machine Learning Emerges (1950s-1990s): Machine learning, a subset of AI, gained prominence during this period. In 1956, Frank Rosenblatt developed the Perceptron, an early neural network model. In the 1980s, researchers like Tom Mitchell and Judea Pearl made significant contributions to the field of machine learning.
Expert Systems Boom (1980s): Expert systems witnessed a boom in the 1980s with commercial applications in various domains, such as medicine and finance. However, the limitations of rule-based systems became apparent, leading to a decline in interest.
Backpropagation Algorithm (1986): In 1986, David Rumelhart, Geoffrey Hinton, and Ronald Williams introduced the backpropagation algorithm, a method for training artificial neural networks. This breakthrough led to renewed interest in neural networks and laid the foundation for future developments in deep learning.
Reinforcement Learning (1990s): Reinforcement learning gained attention with notable advancements, including the development of TD-Gammon (1992), a backgammon-playing program that achieved near-human level performance.
Support Vector Machines (SVM) (1992): The development of Support Vector Machines by Vladimir Vapnik and Alexey Chervonenkis provided a powerful machine learning technique for classification and regression tasks. SVMs became widely used in pattern recognition and data analysis.
1997: IBM's Deep Blue defeats world chess champion Garry Kasparov, showcasing the potential of AI in complex strategic games.
Big Data and Modern Machine Learning (2000s): The 2000s saw the rise of big data (Wikipedia came to existance) and the advent of large-scale machine learning. The availability of massive datasets enabled advancements in areas like natural language processing, computer vision, and recommender systems.
Deep Learning Resurgence (2010s): Deep learning experienced a resurgence with the use of deep neural networks and massive computational power. Notable milestones include the ImageNet competition won by AlexNet (2012) and the development of deep architectures like Google's DeepMind AlphaGo (2016).
ImageNet and Convolutional Neural Networks (CNNs) (2012): The ImageNet Large Scale Visual Recognition Challenge, held annually since 2010, drove advancements in computer vision. The breakthrough in 2012 came with AlexNet, a deep convolutional neural network that significantly improved image classification accuracy.
Natural Language Processing and Word Embeddings (2013): The introduction of word embeddings, such as word2vec and GloVe, enabled more effective representation and understanding of natural language by AI systems. This advancement greatly influenced natural language processing tasks like sentiment analysis and language translation.
GPT-3 and Transformers (2010s): In 2017, the Transformer architecture, introduced by Vaswani et al., revolutionized natural language processing tasks. GPT-3 (2020), a language model developed by OpenAI, showcased the power of large-scale deep learning models in generating human-like text.
Advances in Computer Vision (2010s): Deep learning models achieved remarkable progress in computer vision, leading to breakthroughs in tasks like image recognition, object detection, and image generation. The invention of GANs by Ian Goodfellow introduced a framework for training generative models. Notable models include the development of Faster R-CNN (2015) and Generative Adversarial Networks (GANs).
Reinforcement Learning Breakthroughs (2010s): Reinforcement learning made significant strides with the success of deep reinforcement learning algorithms. Notable achievements include the AlphaGo Zero (2017) and AlphaZero (2018) systems, which surpassed human-level performance in complex games like Go and chess.



Generative AI

Natural Language
Large Language Models, such as GPT-3 and GPT-4, can be considered as a specialized application within the domain of deep learning. They specifically belong to the category of deep learning architectures known as Transformers, which are used for natural language processing and sequence modeling.


