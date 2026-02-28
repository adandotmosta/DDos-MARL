# Flow-Level MARL Router NIDS (PettingZoo + SB3)

This is a minimal scaffold for a flow-level, router-based NIDS environment using PettingZoo's Parallel API and Stable-Baselines3 with parameter sharing.

## Structure
- `envs/flow_router_env.py` Flow-level multi-agent environment
- `scripts/train_ppo.py` PPO training with shared policy
- `requirements.txt` dependencies

## Quick Start
1. Create a venv and install deps
2. Run training:

```bash
python scripts/train_ppo.py --n-agents 3 --total-steps 200000
```

## Environment Summary
- Agents: routers (`router_0 ... router_n`)
- Observations: queue util, CPU load, flow rate, mean packet size, local alerts, neighbor alerts, attack hint
- Actions: allow / rate-limit / deep inspect / block
- Rewards: detect (+), miss (-), false positive (-), overhead (-)

## Next Steps
- Add richer traffic generators and attack stages
- Add communication channels between agents
- Implement attacker adaptation
- Add evaluation metrics scripts



Clarification : 
For simulating NIDS, we need first to define topology, and a dataset so that we can simulate different flows that each router has seen in window based time, cause NIDS operate per window based, so basically each window they see a # number of flows, then they aggregate the flow level statistics( for each flow count number of packets, flow duration, ....), then they use a rule based approach or ML to changge their policy of reaction to the upcoming windows( block, alert, inspect,....).
how to implement our simulator over Line Toplogy :
- defining a time window( 1 ms).
for each time window( RL Step) : 
- we sample from CICDS( or any realistic dataset) N(number of flows) N is generated randomly or over a distribution following attack process( less malicious, more malicous, less malicious : heat then peak then cooldown).
- for each flow we generate a path ( that has a random start over R1, to Rn-1( we may want to more probability to assign a router closer to the end ) ) and fixed destination whcih is (Rn).
- then for each router he aggregates the flows that he only got  received for this time window.
- then each agent act => we calulcate reward and we change policy


question duration flow for each flow is calculated before based on one router, so how to model that in simulaotr for multi routers : 
- each router sees the flow, then the next window that flow is gone( if we assume R1 and R2 see this flow on same window, it might not be realstic since normally here the time flow differ)
- if we model time flow correctly we can change timeflow based on each time we receive,

