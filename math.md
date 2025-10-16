# VI (Categorical) + HMM ELBO — Structured by Components

## Encoder (produces logits and posterior)

At each time step \(t\), the encoder outputs logits $$\ell_t \in \mathbb{R}^K$$ and a categorical posterior:

$$
q_\phi(z_t = k \mid x_{1:T}) = \mathrm{softmax}(\ell_t)_k
\quad\Rightarrow\quad
q_{t,k} = \frac{e^{\ell_{t,k}}}{\sum_{j=1}^K e^{\ell_{t,j}}}.
$$

Collecting over time (mean-field form):

$$
q_\phi(z_{1:T} \mid x_{1:T}) = \prod_{t=1}^T q_\phi(z_t \mid x_{1:T}), 
\qquad
q \in \mathbb{R}^{T \times K}.
$$

---

## Prior (HMM: initial distribution and transitions)

Initial state distribution:

$$
p_\theta(z_1 = k) = \pi_k, 
\qquad 
\sum_{k=1}^K \pi_k = 1.
$$

Transitions (stationary or input-conditioned):

$$
p_\theta(z_t = j \mid z_{t-1} = i,\, u_{1:T}) = A_t[i,j],
\qquad
\sum_{j=1}^K A_t[i,j] = 1 \ \text{for each row } i.
$$

Mean-field expected log prior under \(q\):

$$
\mathbb{E}_q[\log p_\theta(z_{1:T} \mid u_{1:T})]
=
\sum_{k=1}^K q_{1,k}\, \log \pi_k
\;+\;
\sum_{t=2}^T \sum_{i=1}^K \sum_{j=1}^K q_{t-1,i}\, q_{t,j}\, \log A_t[i,j].
$$

Parameterization via logits (normalization):

$$
\pi = \mathrm{softmax}(\alpha), \quad \alpha \in \mathbb{R}^K,
\qquad
A_t[i,\cdot] = \mathrm{softmax}\big(M_t[i,\cdot]\big).
$$

Stationary:
$$
M_t \equiv M \in \mathbb{R}^{K\times K}.
$$

Input-conditioned: 
$$
M_t = g_\theta^{\text{trans}}(u_t) \in \mathbb{R}^{K\times K}.
$$

---

## Decoder (Gaussian emissions via state embedding)

State embedding matrix $$E \in \mathbb{R}^{K \times D}$$ and per-time embedding:

$$
e_t = q_t^\top E \in \mathbb{R}^D,
\qquad
e \in \mathbb{R}^{T \times D}.
$$

Emission parameters (diagonal Gaussian):

$$
(\mu_t,\, \log \sigma_t^2) = g_\theta(e_t),
\qquad
\mu_t,\, \sigma_t \in \mathbb{R}^d.
$$

Per-time log likelihood (diagonal Gaussian):

$$
\log p_\theta(x_t \mid z_t) \approx \log \mathcal{N}\!\big(x_t;\, \mu_t,\, \mathrm{diag}(\sigma_t^2)\big)
= -\tfrac{1}{2}\!\left[
d\log(2\pi)
+ \sum_{j=1}^d \log \sigma^2_{t,j}
+ \sum_{j=1}^d \frac{(x_{t,j}-\mu_{t,j})^2}{\sigma^2_{t,j}}
\right].
$$

Expected reconstruction term under \(q\):

$$
\mathbb{E}_q[\log p_\theta(x_{1:T} \mid z_{1:T})]
=
\sum_{t=1}^T \sum_{k=1}^K q_{t,k}\, \log p_\theta(x_t \mid z_t = k)
\;\;\approx\;\;
\sum_{t=1}^T \log \mathcal{N}\!\big(x_t;\, \mu_t,\, \mathrm{diag}(\sigma_t^2)\big).
$$

(The approximation uses $e_t = q_t^\top E$ to condition $\mu_t,\sigma_t$ directly on $q_t$.)

---

## Main Model (ELBO, entropy, and loss)

Entropy (sum over time and states):

$$
-\mathbb{E}_q[\log q_\phi(z_{1:T} \mid x_{1:T})]
=
\sum_{t=1}^T \sum_{k=1}^K \big(- q_{t,k}\, \log q_{t,k}\big).
$$

Full ELBO:

$$
\mathcal{L}(\theta,\phi)
=
\underbrace{\mathbb{E}_q[\log p_\theta(x_{1:T} \mid z_{1:T})]}_{\text{reconstruction}}
+
\underbrace{\mathbb{E}_q[\log p_\theta(z_{1:T} \mid u_{1:T})]}_{\text{HMM prior}}
-
\underbrace{\mathbb{E}_q[\log q_\phi(z_{1:T} \mid x_{1:T})]}_{\text{entropy}}.
$$

Optional $\beta$-weighting (warm-up):

$$
\mathcal{L}_\beta(\theta,\phi)
=
\mathrm{Recon}
+
\beta \,(\mathrm{Prior} - \mathrm{Entropy}),
\qquad
\beta \in [0,1].
$$

Training objective (minimize negative ELBO):

$$
\mathcal{J}(\theta,\phi) = -\,\mathcal{L}_\beta(\theta,\phi).
$$

---

## Notes on symbols

- $K$: number of discrete hidden states (regimes).
- $T$: number of time steps.
- $d$: data dimension per time step.
- $q_{t,k}$: variational posterior probability of state $k$ at time $t$.
- $\pi$: initial state distribution (with $\sum_{k=1}^K \pi_k = 1$).
- $A_t$: transition probabilities at time $t$ (each row sums to $1$).
- $E \in \mathbb{R}^{K\times D}$: state embedding matrix; $D$ is embedding size.
- $g_\theta$: decoder network producing $(\mu_t,\log\sigma_t^2)$ from $e_t$.
- $u_t$: optional exogenous inputs for input-conditioned transitions.
