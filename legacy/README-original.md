# Quantara

<img width="2257" height="1383" alt="image" src="https://github.com/user-attachments/assets/f139ebd4-b3c8-4d48-a978-6dc9bc14993d" />

**Quantara** is a full-stack financial analytics and trading research platform. It focuses on **portfolio construction**, **risk modeling**, and an **RL trading agent (DDPG)**—with a pragmatic engineering stack (FastAPI + Next.js) and reproducible workflows.

---

## Quick Start

### Backend

```bash
cd ./backend
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8080
```

### Frontend

```bash
cd ./react-web-client
npm i
npm run dev
```

---

## Portfolio Optimization

### Markowitz Mean–Variance

$$
\max_w \; w^T \mu - \frac{\lambda}{2} w^T \Sigma w
\quad \text{s.t.} \quad 1^T w = 1,\; w \in \mathcal{C}
$$

- $w$: portfolio weights  
- $\mu$: expected returns  
- $\Sigma$: covariance matrix  
- $\lambda$: risk aversion  

---

### Risk Parity (ERC)

$$
RC_i(w) = \frac{w_i (\Sigma w)_i}{\sqrt{w^T \Sigma w}}
$$

Equalize all $RC_i$:

$$
\min_w \sum_i \left( RC_i(w) - \bar{RC} \right)^2
\quad \text{s.t.} \quad 1^T w = 1,\; w \ge 0
$$

---

### Black–Litterman

$$
\mu_{BL} = \Big[(\tau \Sigma)^{-1} + P^T \Omega^{-1} P\Big]^{-1}
\Big[(\tau \Sigma)^{-1}\pi + P^T \Omega^{-1}q\Big]
$$

---

### CVaR (Rockafellar–Uryasev)

$$
\min_{w, \zeta, u_k} \zeta + \frac{1}{(1-\alpha)K} \sum_{k=1}^K u_k
$$

subject to:

$$
u_k \ge L^{(k)}(w) - \zeta, \quad u_k \ge 0, \quad 1^T w = 1
$$

---

## Reinforcement Learning (DDPG)

### MDP Setup
- **State $s_t$**: prices, holdings, cash  
- **Action $a_t$**: portfolio allocation  
- **Reward $r_t$**: log-return with costs  

### Critic Loss

$$
\mathcal{L}_Q = \Big(Q_\theta(s_t,a_t) -
(r_t + \gamma Q_{\theta^-}(s_{t+1},\pi_{\phi^-}(s_{t+1})))\Big)^2
$$

### Actor Gradient

$$
\nabla_\phi J \approx
\mathbb{E}_s \Big[ \nabla_a Q_\theta(s,a)\rvert_{a=\pi_\phi(s)} \;
\nabla_\phi \pi_\phi(s) \Big]
$$

### Stability Tricks
- Target networks (Polyak averaging):  
  $\theta^- \leftarrow \tau \theta + (1-\tau)\theta^-$  
- Ornstein–Uhlenbeck noise for exploration  
- Replay buffer for experience sampling  

---

## Metrics

- **Sharpe Ratio**:  
  $Sharpe = \frac{E[R_p] - r_f}{\sigma_p}$  

- **Max Drawdown**:  
  $MDD = \max_{t} \Big( \frac{\text{Peak}_t - \text{Val}_t}{\text{Peak}_t} \Big)$  

- **CVaR**: conditional expectation of loss beyond VaR  

---
