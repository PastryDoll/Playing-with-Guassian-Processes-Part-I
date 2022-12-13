# Playing-with-Guassian-Processes-Part-I

We will divide this study in 2 parts. The second part will deal with Neural Networks and hyperparameters. (Some of this code is based on the code from https://peterroelants.github.io/posts/gaussian-process-tutorial/)

This code is a study on Gaussian Processes and GP Bayesian Regression. We analize the theory behind GPs, and we also implement GP Bayesian Regression for a real life situation of trying to find kids in a cave.

Let $\phi(x) : x\in \mathbb{R}^d$.
The joint Prob. Density for a set of point $(t_1, \dots, t_n)$ is :

$p(\phi(t_1), \phi(t_2), \dots, \phi(t_n))$.

For a Gaussian Randon Field, $p$ is Gaussian :

$p(\phi(t_1), \phi(t_2), \dots, \phi(t_n)) = \frac{1}{\sqrt{(2\pi)^N|detC|}}exp[-\frac{1}{2}(\phi - \bar{\phi})^TC^{-1}(\phi - \bar{\phi})]$, with

$\bar{\phi_j} = <\phi_j>$, and
$C_{lj} = <(\phi - \bar{\phi})_l(\phi - \bar{\phi})_j>_p$

If we take a measurement $y_j$ at $j$, and assuming this is a Gaussian (measurement noise):

$p(y_j|\phi) = \frac{1}{\sqrt{2\pi}\sigma}exp[-\frac{1}{2\sigma^2}(y_j - \phi_j)^2]$.

Applying Bayes:

$p(\phi|y_j) = \frac{p(y_j|\phi)p(\phi)}{p(y_j)} = \frac{p(y_j|\phi)p(\phi)}{Normalization} => $ 
$log(p(\phi|y_j)) = cont - \frac{1}{2}\delta\phi^TC'^{-1}\delta\phi + (y_j - \bar{\phi}_j)\frac{\delta\phi_j}{\sigma^2}$

The new values of $C$, and $\bar{\phi}$ are

$\bar{\phi}' = \bar{\phi} + C'\ket{j}\frac{y_j - \bar{\phi_j}}{\sigma^2}$

$C'^{-1} = C^{-1} + \frac{1}{\sigma^2}\ket{j}\bra{j}$, where $(\ket{j})_l = \delta_{jl}$ 

We will make some new notations that will make it easier to define $C'$ for a finite set of measures $Y$. Let $Y$ be a set of new measures, $X$ be the old set of measures defining $C$, and lets take the joint distribution on the $(X,Y)$, where $C$ is the matrix with $C_{YY}, C_{XX}$ on the diagonals, and $C_{XY}, C_{YX}$ on the off diagonals. Then 

$\bar{\phi}_{X|Y} = \bar{\phi}_{X} + C_{XY}C^{-1}_{YY}(Y - \bar{\phi}_Y)$

$C_{X|Y} = C_{XX} - C_{XY}C^{-1}_{YY}C_{YX}$.

If we have a measument noise we have to add to $C_{YY}$  something like $\sigma^2\mathbb{I}$

We can further write

$C_{X|Y} = C_{XX} - (C^{-1}_{YY}C_{YX})^{T}C_{YX}$

$\bar{\phi}_{X|Y} = \bar{\phi}_{X} + (C_{YY}^{-1}C_{YX})^{T}(Y - \bar{\phi}_Y)$

![Bayes Regression](bayes.png?raw=true "GP Bayes Regression 1D")
![Exponential Quadratic Kernel](expqua.png?raw=true "Exponential Quadratic Kernel")

Bayes Regression 2D:

![Bayes Regression 2D](bayes2d.png?raw=true "GP Bayes Regression 2D")

